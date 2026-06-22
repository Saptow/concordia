import threading
import time
import unittest

from concordia.hdb_simulation import listing_portal
from concordia.utils import concurrency


class _ConcurrentWriteDetectingClient:

  def __init__(self):
    self._lock = threading.Lock()
    self._active_set_payload_calls = 0
    self.max_concurrent_set_payload_calls = 0

  def collection_exists(self, collection_name: str) -> bool:
    del collection_name
    return True

  def set_payload(self, *, collection_name, payload, points) -> None:
    del collection_name, payload, points
    with self._lock:
      self._active_set_payload_calls += 1
      self.max_concurrent_set_payload_calls = max(
          self.max_concurrent_set_payload_calls,
          self._active_set_payload_calls,
      )
    try:
      time.sleep(0.02)
    finally:
      with self._lock:
        self._active_set_payload_calls -= 1


class ListingPortalRetrieverThreadSafetyTest(unittest.TestCase):

  def test_concurrent_payload_updates_are_serialized(self):
    client = _ConcurrentWriteDetectingClient()
    retriever = listing_portal.ListingPortalRetriever(
        client=client,
        collection_name='test_collection',
    )

    _, errors = concurrency.run_tasks_in_background({
        'seller_1': lambda: retriever.deactivate_listing('seller_1'),
        'seller_2': lambda: retriever.deactivate_listing('seller_2'),
    })

    self.assertEqual(errors, {})
    self.assertEqual(client.max_concurrent_set_payload_calls, 1)


if __name__ == '__main__':
  unittest.main()
