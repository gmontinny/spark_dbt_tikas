-- gold/mart_document_summaries.sql
-- depends_on: {{ ref('stg_documents_enriched') }}
-- Tabela de sumarizações geradas pelo LLM, pronta para RAG e busca semântica.
-- Armazenada no StarRocks para consulta de baixa latência.

with base as (
    select * from {{ ref('stg_documents_enriched') }}
    where summary is not null and length(summary) > 20
)

select
    file_name,
    document_title,
    document_author,
    topic,
    language,
    summary,
    text_length,
    num_pages,
    enriched_at
from base
order by enriched_at desc
