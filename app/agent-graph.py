from __future__ import annotations

import json

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticToolsParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from app.api import get_catalog
from app.config import Settings, get_settings
from app.models import BuilderResult, SQLState
from app.schema_loader import SchemaCatalog, load_schema_catalog
from app.sql_builder import SqlBuilder
from app.sql_revisor import SqlRevisor
from app.sql_validator import SqlValidator

settings: Settings = Settings()

schema_catalog: SchemaCatalog = load_schema_catalog(settings.schema_file, settings.allowed_schemas)

client = ChatGoogleGenerativeAI(
    google_api_key=settings.gemini_api_key,
    model=settings.gemini_model,
    temperature=0,
)

sql_builder = SqlBuilder(get_settings(), get_catalog())

sql_validator = SqlValidator(get_settings(), get_catalog())

sql_revisor = SqlRevisor(get_settings(), get_catalog())


def build_sql(state: SQLState) -> SQLState:
    """This node builds SQL from a requirement using the SQL builder."""
    builder_result = sql_builder.build(state.requirement)
    state.sql = builder_result.sql
    state.explanation = builder_result.explanation
    state.tables_used = builder_result.tables_used
    state.assumptions = builder_result.assumptions
    return state

def validate_sql(state: SQLState) -> SQLState:
    """This node validates SQL using the SQL validator."""
    validation_result = sql_validator.validate(state.sql)
    state.is_valid = validation_result.is_valid
    state.sanitized_sql = validation_result.sanitized_sql
    state.errors.extend(validation_result.errors)
    state.warnings.extend(validation_result.warnings)
    state.tables_used.extend(validation_result.tables_used)
    return state

def revise_sql(state: SQLState) -> SQLState:
    """This node revises SQL using the SQL revisor."""
    revised_state = sql_revisor.revise(state)
    return revised_state