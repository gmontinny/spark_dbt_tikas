
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    

select
    file_name as unique_field,
    count(*) as n_records

from `gold`.`stg_documents_enriched`
where file_name is not null
group by file_name
having count(*) > 1



  
  
      
    ) dbt_internal_test