
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select file_name
from `gold`.`mart_documents_by_topic`
where file_name is null



  
  
      
    ) dbt_internal_test