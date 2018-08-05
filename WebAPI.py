import requests
import json
data =  "{\"input_df\": [{\"TimeSpentOnWeb\": 5.1, \"TimeSpentOnProductPage\": 3.5}]}"

url='http://23.101.154.86/api/v1/service/productclusterapp/score'
api_key = 'p1oZo220XYGcu2OzbPX5kJcKdASVL5hu' 
headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

resp = requests.post(url, data, headers=headers)
print(resp.text)
## az login
## az env set -n dscluster -g datascience
## az ml service list realtime -o table
#az ml service usage realtime -i productclusterapp.dscluster-bcd11878.eastus2
#az ml service keys realtime -i productclusterapp.dscluster-bcd11878.eastus2