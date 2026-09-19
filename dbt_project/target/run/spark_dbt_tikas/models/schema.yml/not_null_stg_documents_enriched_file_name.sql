
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select file_name
from `gold`.`stg_documents_enriched`
where file_name is null



  
  
      
    ) dbt_internal_test