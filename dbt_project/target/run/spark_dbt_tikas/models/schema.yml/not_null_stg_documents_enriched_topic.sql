
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select topic
from `gold`.`stg_documents_enriched`
where topic is null



  
  
      
    ) dbt_internal_test