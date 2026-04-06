"""
Agent ecosystem for forecasting pipeline.

Pipeline agents (BaseAgent interface — execute(ctx) -> (ctx, decision)):
  - MemoryManagerAgent: Pre-flight resource check
  - DataAnalyzerAgent: Data profiling and cleaning recommendations
  - ExternalDataAgent: External data discovery and joins
  - FeatureEngineerAgent: Feature specification generation
  - ModelSelectorAgent: Deterministic model selection (rules engine)

Chat & tools:
  - ChatAgentV2: Conversational interface with OpenAI function calling
  - ToolRegistry: Extensible tool registry (30 tools, 7 categories)
  - DataOperations: DataFrame transformation engine

Legacy agents:
  - ForecastingPlanner: LLM-powered planner
"""

from forecaster.agents.base import AgentResponse, BaseAgent
from forecaster.agents.data_analyzer import DataAnalyzerAgent
from forecaster.agents.external_data_agent import ExternalDataAgent
from forecaster.agents.feature_engineer import FeatureEngineerAgent
from forecaster.agents.memory_manager import MemoryManagerAgent
from forecaster.agents.model_selector import ModelSelectorAgent
from forecaster.agents.planner import ForecastingPlanner

__all__ = [
    "BaseAgent",
    "AgentResponse",
    "DataAnalyzerAgent",
    "ExternalDataAgent",
    "FeatureEngineerAgent",
    "MemoryManagerAgent",
    "ModelSelectorAgent",
    "ForecastingPlanner",
]
