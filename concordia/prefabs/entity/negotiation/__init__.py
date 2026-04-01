"""Negotiation prefab package with lazy module loading."""

from importlib import import_module

from concordia.prefabs.entity.negotiation.constants import (
    DEFAULT_MODULE_CONFIGS,
    MODULE_COMPONENT_NAMES,
    ModuleType,
)
from concordia.prefabs.entity.negotiation.config import (
    AlgorithmConfig,
    DeceptionDetectionConfig,
    EvaluationConfig,
    InterpretabilityConfig,
    ModuleDefaults,
    OutcomeConfig,
    ParsingConfig,
    RelationshipConfig,
    StrategyConfig,
    TheoryOfMindConfig,
)

_NEGOTIATOR_MODULES = {
    'base_negotiator': '.base_negotiator',
    'advanced_negotiator': '.advanced_negotiator',
    'uncertain_negotiator': '.uncertain_negotiator',
}

__all__ = [
    'base_negotiator',
    'advanced_negotiator',
    'uncertain_negotiator',
    'build_agent',
    'build_advanced_agent',
    'build_custom_agent',
    'BaseNegotiator',
    'AdvancedNegotiator',
    'CustomNegotiator',
    'ModuleType',
    'MODULE_COMPONENT_NAMES',
    'DEFAULT_MODULE_CONFIGS',
    'StrategyConfig',
    'OutcomeConfig',
    'AlgorithmConfig',
    'ModuleDefaults',
    'TheoryOfMindConfig',
    'DeceptionDetectionConfig',
    'EvaluationConfig',
    'InterpretabilityConfig',
    'RelationshipConfig',
    'ParsingConfig',
]


def __getattr__(name: str):
    if name in _NEGOTIATOR_MODULES:
        return import_module(_NEGOTIATOR_MODULES[name], __name__)
    if name == 'build_agent':
        return import_module('.base_negotiator', __name__).build_agent
    if name == 'build_advanced_agent':
        return import_module('.advanced_negotiator', __name__).build_agent
    if name == 'build_custom_agent':
        return import_module('.uncertain_negotiator', __name__).build_agent
    if name == 'BaseNegotiator':
        return import_module('.base_negotiator', __name__).Entity
    if name == 'AdvancedNegotiator':
        return import_module('.advanced_negotiator', __name__).Entity
    if name == 'CustomNegotiator':
        return import_module('.uncertain_negotiator', __name__).Entity
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
