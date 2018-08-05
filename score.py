# This script generates the scoring and schema files
# necessary to operationalize your model
from azureml.api.schema.dataTypes import DataTypes
from azureml.api.schema.sampleDefinition import SampleDefinition
from azureml.api.realtime.services import generate_schema
from azureml.assets import get_local_path
import numpy as np
import pandas as pd
from keras.models import load_model

# Prepare the web service definition by authoring
# init() and run() functions. Test the functions
# before deploying the web service.

model = None

def init():
    # Get the path to the model asset
    # local_path = get_local_path('mymodel.model.link')
    
    # Load model using appropriate library and function
    #global model
    # model = model_load_function(local_path)
    #from sklearn.externals import joblib
    global model
    #model=joblib.load('PredictPurchase_model.h5')
    model= load_model("PredictPurchase_model.h5")

def run(input_df):
    import json
    
    # Predict using appropriate functions
    # prediction = model.predict(input_df)
    y_pred = model.predict_classes(input_df)
    predictions=pd.DataFrame(data=y_pred,columns=['PredictedProduct'])
    predictions['PredictedProduct']= predictions['PredictedProduct'].apply(lambda x: 'Product_C' if x ==2 else ('Product_B' if x==1 else 'Product_A'))
    return json.dumps(str(predictions['PredictedProduct'][0]))

    #prediction = "%s %d" % (str(input_df), model)
    #return json.dumps(str(prediction))

#def generate_api_schema():
#    import os
#    import pandas as pd
#    columns=['TimeSpentOnWeb','TimeSpentOnProductPage']
#    sample_input = pd.DataFrame(data=[(5.1,3.5)],columns=columns)
#    print("create schema")
#    #sample_input = "sample data text"
#    inputs = {"input_df": SampleDefinition(DataTypes.PANDAS, sample_input)}
#    os.makedirs('outputs', exist_ok=True)
#    print(generate_schema(inputs=inputs, filepath="service-schema.json", run_func=run))

# Implement test code to run in IDE or Azure ML Workbench
if __name__ == '__main__':
    # Import the logger only for Workbench runs
    from azureml.logging import get_azureml_logger
    import numpy as np
    import pandas as pd
    columns=['TimeSpentOnWeb','TimeSpentOnProductPage']
    sample_input = pd.DataFrame(data=[(5.1,3.5)],columns=columns)

    init()
    #input = "{}"
    result = run(sample_input)
    print("The predicted Product which will be Purchased is -",str(result))

    ##Generating Schema
    inputs = {"input_df": SampleDefinition(DataTypes.PANDAS, sample_input)}
    print(generate_schema(inputs=inputs, filepath="service-schema.json", run_func=run))

    logger = get_azureml_logger()
    logger.log("Result",result)

    ## To use your existing model management account
    ## az ml account modelmanagement set -n dsazmlexpmodelmgmt -g datascience
    ## Setup a NEW Model management
    ##az ml account modelmanagement create -l [Azure region, e.g. eastus2] -n [your account name] -g [resource group name] --sku-instances [number of instances, e.g. 1] --sku-name [Pricing tier for example S1]
    ## Next setup your local or cluster mode deployment
    ## az ml env setup -l eastus2 -n dslocal -g datascience
    ## az ml env show -l eastus2 -n dslocal -g datascience  
    ## az ml env set -n dslocal -g datascience  
    ## Alternatively For setting up ACS-Kubernetes Cluster mode, create 3 agent  (default)
    ## az ml env setup -l eastus2 -n dscluster -g datascience --cluster --agent-count-3
    ## az ml env set -n dscluster -g datascience  
    ## Next once the envoriment is set, the next command in one step will register model, create manifest, create image and setup service
    ## command - az ml service create realtime -f score.py --model-file PredictPurchase_model.h5 
    # -s service-schema.json -n productapp -r python--collect-model-data true -c aml_config\conda_dependencies.yml
    ## Alternatively you can do this step by step and keep passing each Ids to the next step 
    ##az ml service create realtime -n productapp -r python --collect-model-data true --image-id 8f14e874-4473-40d7-aca2-f2a651d24c48
    ## Example:
    ##az ml model register --model model.pkl --name <<GiveYourModelName>> (This will return a ModelId, next we build a Manifest)
    ##az ml manifest create --manifest-name <<GiveAManifestName>> -f score.py -r python -s service_schema.json -c aml_config/conda_dependencies.yml --model-id <ModelId> (Next We Need to Create Image)
    ##az ml image create --image-name <<GiveAnImageName>> -f score.py -r python -s service_schema.json -c aml_config/conda_dependencies.yml --manifest-id <<ManifestId from the previous step>>
    ##Az ml service create realtime -n <<GiveServiceAName>> -r python --collect-model-data true --image-id <<Image Id from Previous Step>>
    ## Now using the Service
    ## az ml service run realtime -i productapp -d "{\"input_df\": [{\"TimeSpentOnWeb\": 5.1, \"TimeSpentOnProductPage\": 3.5}]}"
    ##az ml service logs realtime -i <<ServiceName>> (For getting the service telemetry)
    ##https://docs.microsoft.com/en-us/azure/machine-learning/desktop-workbench/how-to-read-write-files
    ##https://docs.microsoft.com/en-us/azure/azure-resource-manager/resource-manager-register-provider-errors
    ##az ml env setup -n dscluster -g datascience -l eastus2 --cluster --agent-count 2
    ##az ml env delete -n dscluster -g datascience 

    
