create view `gold`.`stg_documents_enriched` as -- staging/stg_documents_enriched.sql
-- Lê os documentos enriquecidos pela inferência LLM gravados no StarRocks
-- e normaliza para consumo pelos modelos Gold do dbt.

select
    file_name,
    coalesce(nullif(trim(title), ''), file_name)       as document_title,
    coalesce(nullif(trim(author), ''), 'Desconhecido') as document_author,
    nullif(trim(language), '')                         as language,
    cast(nullif(trim(num_pages), '') as int)            as num_pages,
    cast(text_length as int)                            as text_length,
    trim(topic)                                        as topic,
    trim(summary)                                      as summary,
    content_type,
    ingested_at,
    processed_at,
    enriched_at
from gold.documents_enriched
where file_name is not null;