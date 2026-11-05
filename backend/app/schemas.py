from pydantic import BaseModel, ConfigDict, model_validator


class SchemaMappingResult(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    part_id: str | None
    transaction_date: str | None
    quantity_sold: str | None
    part_name: str | None
    category: str | None
    current_stock: str | None
    lead_time_days: str | None
    unit_cost: str | None

    @model_validator(mode='after')
    def unique_sources(self):
        values = [v for v in self.model_dump().values() if v is not None]
        if len(values) != len(set(values)):
            raise ValueError('Each source column may be assigned only once')
        return self


FIELDS = list(SchemaMappingResult.model_fields)
REQUIRED = ['part_id', 'transaction_date', 'quantity_sold']
