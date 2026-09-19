
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select file_name
from `gold`.`mart_document_summaries`
where file_name is null



  
  
      
    ) dbt_internal_test