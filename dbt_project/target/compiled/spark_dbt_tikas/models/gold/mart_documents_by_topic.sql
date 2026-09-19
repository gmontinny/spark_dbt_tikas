-- gold/mart_documents_by_topic.sql
-- depends_on: `gold`.`stg_documents_enriched`
-- Tabela analítica: documentos agrupados por tópico com métricas de conteúdo.
-- Consumida pelo StarRocks + Superset.

with base as (
    select * from `gold`.`stg_documents_enriched`
),

topic_stats as (
    select
        topic,
        count(*)                            as total_documents,
        avg(text_length)                    as avg_text_length,
        avg(coalesce(num_pages, 1))         as avg_pages,
        min(ingested_at)                    as first_ingested_at,
        max(ingested_at)                    as last_ingested_at
    from base
    group by topic
)

select
    b.file_name,
    b.document_title,
    b.document_author,
    b.language,
    b.num_pages,
    b.text_length,
    b.topic,
    b.summary,
    b.content_type,
    b.ingested_at,
    b.enriched_at,
    ts.total_documents       as topic_total_docs,
    ts.avg_text_length       as topic_avg_text_length,
    ts.avg_pages             as topic_avg_pages
from base b
left join topic_stats ts on b.topic = ts.topic