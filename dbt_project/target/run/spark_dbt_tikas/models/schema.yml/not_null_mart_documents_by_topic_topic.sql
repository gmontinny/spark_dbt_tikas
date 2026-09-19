
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select topic
from `gold`.`mart_documents_by_topic`
where topic is null



  
  
      
    ) dbt_internal_test