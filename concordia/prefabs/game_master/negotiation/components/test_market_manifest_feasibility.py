import json
import os
from pathlib import Path
import unittest

from configs import REPO_ROOT, SegmentConfig


def _default_manifest_path() -> Path:
  override = os.environ.get('HDB_TEST_MARKET_MANIFEST', '').strip()
  if override:
    return Path(override)
  return REPO_ROOT / SegmentConfig().output_dir / 'manifest.json'


def _resolve_manifest_artifact_path(
    manifest_file: Path,
    raw_path: str | Path,
) -> Path:
  candidate = Path(str(raw_path))
  if candidate.is_absolute():
    return candidate.resolve()

  manifest_relative = (manifest_file.parent / candidate).resolve()
  if manifest_relative.exists():
    return manifest_relative

  repo_relative = (REPO_ROOT / candidate).resolve()
  if repo_relative.exists():
    return repo_relative

  return manifest_relative


def _read_jsonl_records(path: Path) -> list[dict[str, object]]:
  return [
      json.loads(line)
      for line in path.read_text(encoding='utf-8').splitlines()
      if line.strip()
  ]


def _load_bundle_from_manifest(
    manifest_path: Path,
) -> tuple[dict[str, object], dict[str, Path]]:
  manifest_file = manifest_path.resolve()
  manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
  required_keys = (
      'flat_units_path',
      'sellers_path',
      'buyers_broad_path',
      'buyers_retained_path',
  )
  missing_keys = [
      key for key in required_keys if not str(manifest.get(key, '')).strip()
  ]
  if missing_keys:
    raise ValueError(
        f'Manifest {manifest_file} is missing required path keys: {missing_keys}'
    )

  resolved_manifest = {
      key: _resolve_manifest_artifact_path(manifest_file, manifest[key])
      for key in required_keys
  }
  bundle = {
      'flats': _read_jsonl_records(resolved_manifest['flat_units_path']),
      'sellers': _read_jsonl_records(resolved_manifest['sellers_path']),
      'buyers_broad': _read_jsonl_records(resolved_manifest['buyers_broad_path']),
      'buyers_retained': _read_jsonl_records(
          resolved_manifest['buyers_retained_path']
      ),
  }
  return bundle, resolved_manifest


def _dedupe_preserving_order(values: list[str]) -> list[str]:
  seen: set[str] = set()
  ordered: list[str] = []
  for value in values:
    if not value or value in seen:
      continue
    seen.add(value)
    ordered.append(value)
  return ordered


def _candidate_buyer_ids_for_seller(
    seller: dict[str, object],
    buyers_by_id: dict[str, dict[str, object]],
) -> list[str]:
  linked_flat_id = str(seller.get('linked_flat_id', '')).strip()

  candidate_buyer_ids: list[str] = []
  seeded_buyer_id = str(seller.get('seeded_buyer_id', '')).strip()
  if seeded_buyer_id:
    candidate_buyer_ids.append(seeded_buyer_id)
  candidate_buyer_ids.extend(
      str(buyer_id).strip()
      for buyer_id in seller.get('potential_buyer_ids', ())
      if str(buyer_id).strip()
  )

  if not candidate_buyer_ids:
    for buyer_id, buyer in buyers_by_id.items():
      feasible_flat_ids = {
          str(flat_id).strip()
          for flat_id in buyer.get('feasible_flat_ids', ())
          if str(flat_id).strip()
      }
      if linked_flat_id and linked_flat_id not in feasible_flat_ids:
        continue
      candidate_buyer_ids.append(buyer_id)

  filtered_candidates: list[str] = []
  for buyer_id in _dedupe_preserving_order(candidate_buyer_ids):
    buyer = buyers_by_id.get(buyer_id)
    if buyer is None:
      continue
    feasible_flat_ids = {
        str(flat_id).strip()
        for flat_id in buyer.get('feasible_flat_ids', ())
        if str(flat_id).strip()
    }
    if linked_flat_id and linked_flat_id not in feasible_flat_ids:
      continue
    filtered_candidates.append(buyer_id)
  return filtered_candidates


def _maximum_bipartite_matching(
    seller_to_buyer_ids: dict[str, list[str]],
) -> dict[str, str]:
  buyer_to_seller: dict[str, str] = {}

  def _try_match(seller_id: str, seen_buyers: set[str]) -> bool:
    for buyer_id in seller_to_buyer_ids.get(seller_id, ()):
      if buyer_id in seen_buyers:
        continue
      seen_buyers.add(buyer_id)
      current_seller = buyer_to_seller.get(buyer_id)
      if current_seller is None or _try_match(current_seller, seen_buyers):
        buyer_to_seller[buyer_id] = seller_id
        return True
    return False

  for seller_id in sorted(
      seller_to_buyer_ids,
      key=lambda candidate: (
          len(seller_to_buyer_ids.get(candidate, ())),
          candidate,
      ),
  ):
    _try_match(seller_id, set())

  return {
      seller_id: buyer_id for buyer_id, seller_id in buyer_to_seller.items()
  }


class MarketManifestFeasibilityTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.manifest_path = _default_manifest_path()
    if not cls.manifest_path.exists():
      raise unittest.SkipTest(
          'No processed market manifest found at '
          f'{cls.manifest_path}. Set HDB_TEST_MARKET_MANIFEST to override.'
      )
    cls.bundle, cls.resolved_manifest = _load_bundle_from_manifest(
        cls.manifest_path
    )
    cls.buyers_by_id = {
        str(buyer['buyer_id']): buyer
        for buyer in cls.bundle['buyers_retained']
        if str(buyer.get('buyer_id', '')).strip()
    }
    cls.sellers_by_id = {
        str(seller['seller_id']): seller
        for seller in cls.bundle['sellers']
        if str(seller.get('seller_id', '')).strip()
    }
    cls.seller_to_candidates = {
        seller_id: _candidate_buyer_ids_for_seller(seller, cls.buyers_by_id)
        for seller_id, seller in cls.sellers_by_id.items()
    }

  def test_every_seller_has_at_least_one_feasible_retained_buyer(self):
    sellers_without_candidates = sorted(
        seller_id
        for seller_id, candidate_buyer_ids in self.seller_to_candidates.items()
        if not candidate_buyer_ids
    )
    self.assertFalse(
        sellers_without_candidates,
        (
            f'Manifest {self.manifest_path} contains sellers with no feasible '
            f'retained buyers: {sellers_without_candidates[:10]}'
        ),
    )

  def test_manifest_admits_full_seller_to_buyer_matching(self):
    matching = _maximum_bipartite_matching(self.seller_to_candidates)
    unmatched_sellers = sorted(
        seller_id
        for seller_id in self.sellers_by_id
        if seller_id not in matching
    )
    if unmatched_sellers:
      candidate_summary = {
          seller_id: self.seller_to_candidates.get(seller_id, [])[:5]
          for seller_id in unmatched_sellers[:10]
      }
      self.fail(
          f'Manifest {self.manifest_path} does not admit a full seller-to-buyer '
          f'matching across retained buyers. Unmatched sellers: '
          f'{unmatched_sellers[:10]}. Candidate buyers sample: {candidate_summary}'
      )

    self.assertEqual(len(matching), len(self.sellers_by_id))


if __name__ == '__main__':
  unittest.main()
