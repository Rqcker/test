import ipywidgets as widgets
from IPython.display import display as displayWidget
import json
import uuid
import rdflib
import urllib.request, json 
from google.colab import files
import copy
from pyshacl import validate
from rdflib import URIRef, BNode, Literal
from rdflib import Graph

base_iri= 'http://example.com/'
accountableAgent = {}
accountableAgent ['@id'] = base_iri+ str(uuid.uuid4());

#create json-ld context fro our graph
context = {}
#RDFS
context ['rdfs:comment'] = "http://www.w3.org/2000/01/rdf-schema#comment";
context ['rdfs:label'] = "http://www.w3.org/2000/01/rdf-schema#label";
context ['rdfs:seeAlso'] = "http://www.w3.org/2000/01/rdf-schema#seeAlso";
#OWL context
context['owl:NamedIndividual'] = "http://www.w3.org/2002/07/owl#NamedIndividual";
#EP-PLAN
context['ep-plan:MultiActivity'] = "https://w3id.org/ep-plan#MultiActivity";
context['ep-plan:correspondsToVariable'] = {"@id":"https://w3id.org/ep-plan#correspondsToVariable","@type": "@id"};
context['ep-plan:correspondsToStep'] = {"@id":"https://w3id.org/ep-plan#correspondsToStep","@type": "@id"};

#MLS context
context['mls:Dataset'] = "http://www.w3.org/ns/mls#Dataset";
context['mls:Model'] = "http://www.w3.org/ns/mls#Model";
context['mls:ModelEvaluation'] = "http://www.w3.org/ns/mls#ModelEvaluation";
context['mls:Algorithm'] = "http://www.w3.org/ns/mls#Algorithm";
context['mls:DatasetCharacteristic']="http://www.w3.org/ns/mls#DatasetCharacteristic";
#RAINS context
context['rains:Risk'] = "https://w3id.org/rains#Risk";
context['rains:Bias'] = "https://w3id.org/rains#Bias";
context['rains:Limitation'] = "https://w3id.org/rains#Limitation";

context['rains:IncorrectUseCase'] = "https://w3id.org/rains#IncorrectUseCase";
context['rains:IntendedUseCase'] = "https://w3id.org/rains#IntendedUseCase";
context['rains:IntendedUserGroup'] = "https://w3id.org/rains#IntendedUserGroup";
context['rains:EvaluationResult'] = "https://w3id.org/rains#EvaluationResult";
context['rains:EvaluationMeasure'] = "https://w3id.org/rains#EvaluationMeasure";
context['rains:RealizableObject'] = "https://w3id.org/rains#RealizableObject";
context['rains:RealizedObject'] = "https://w3id.org/rains#RealizedObject";
context['rains:Limitation'] = "https://w3id.org/rains#Limitation";
context['rains:Tradeoff'] = "https://w3id.org/rains#Tradeoff";
context['rains:version'] = "https://w3id.org/rains#version";
context['rains:versionNote'] = "https://w3id.org/rains#versionNote";
context['rains:computedOnSlice'] = "https://w3id.org/rains#computedOnSlice";
context['rains:computedOnDecisionThreshold'] = "https://w3id.org/rains#computedOnDecisionThreshold";
context['rains:hasResultValue'] = "https://w3id.org/rains#hasResultValue";
context['rains:hasResultUpperBound'] = "https://w3id.org/rains#hasResultUpperBound";
context['rains:hasResultLowerBound'] = "https://w3id.org/rains#hasResultLowerBound";



#need to handle proper xsd format - to do later
context['rains:modelInputFormat'] = "https://w3id.org/rains#modelInputFormat";
context['rains:modelOutputFormat'] = "https://w3id.org/rains#modelOutputFormat";
context['rains:versionDate'] = "https://w3id.org/rains#versionDate";
context['rains:hasMitigationStrategy'] = "https://w3id.org/rains#hasMitigationStrategy";
context['rains:hasBase64Image'] = "https://w3id.org/rains#hasBase64Image";
context['rains:isEvaluationResultOf'] = {"@id":"https://w3id.org/rains#isEvaluationResultOf","@type": "@id"};
context['rains:hasRealizableObjectCharacteristic'] = {"@id":"https://w3id.org/rains#hasRealizableObjectCharacteristic","@type": "@id"};
#SAO CONTEXT
context ['sao:InformationRealization']   = "https://w3id.org/sao#InformationRealization";
context ['sao:InformationElement']   = "https://w3id.org/sao#InformationElement";
context ['sao:AccountableAgent']   = "https://w3id.org/sao#AccountableAgent";
context ['sao:isAccountableFor']   = {"@id":"https://w3id.org/sao#isAccountableFor","@type": "@id"};
#PROV CONTEXT
context ['prov:wasDerivedFrom']   = {"@id":"http://www.w3.org/ns/prov#wasDerivedFrom","@type": "@id"};
context ['prov:hadMember']   = {"@id":"http://www.w3.org/ns/prov#hadMember","@type": "@id"};
context ['prov:wasMemberOf']   = {"@id":"http://www.w3.org/ns/prov#wasMemberOf","@type": "@id"};
context ['prov:wasGeneratedBy']   = {"@id":"http://www.w3.org/ns/prov#wasGeneratedBy","@type": "@id"};
context ['prov:used']   = {"@id":"http://www.w3.org/ns/prov#used","@type": "@id"};
context ['prov:wasAssociatedWith']   = {"@id":"http://www.w3.org/ns/prov#wasAssociatedWith","@type": "@id"};
#DC context 
context['dc:LicenseDocument'] = "http://purl.org/dc/terms/LicenseDocument"



#model card captures data about three elements model, training dataset, and evaluation dataset. All the metadata relates to these three high level concepts 
def createInformationElement ():
  el ={}
  el ['@id'] = base_iri+ str(uuid.uuid4());
  el ['@type'] = [];
  el ['@type'].append (context['owl:NamedIndividual']);
  #el ['@type'].append (context['sao:InformationElement']);
  return el

def initializeMappingTool ():

 global dataTransformActivity
 dataTransformActivity = {}
 dataTransformActivity ['@id'] = base_iri+ str(uuid.uuid4());
 dataTransformActivity ['@type'] = [];
 dataTransformActivity ['@type'].append(context['ep-plan:MultiActivity']);
 dataTransformActivity ['@type'].append (context['owl:NamedIndividual']);
 dataTransformActivity ['prov:wasAssociatedWith'] = [];
 dataTransformActivity ['rdfs:comment'] = "Auto generated text: This activity describes the process of producing training and/or evaluation datasets.";

 global trainingDatasetInfoElement
 trainingDatasetInfoElement =createInformationElement ();
 trainingDatasetInfoElement ['@type'].append(context['mls:Dataset']);
 trainingDatasetInfoElement ['@type'].append (context['rains:RealizedObject']);
 trainingDatasetInfoElement ['@type'].append (context['rains:RealizableObject']);
 trainingDatasetInfoElement ['rains:hasRealizableObjectCharacteristic'] = [];
 
 
 global trainingDatasetInformationRealization
 trainingDatasetInformationRealization ={}
 trainingDatasetInformationRealization ['@id'] = base_iri+ str(uuid.uuid4());
 trainingDatasetInformationRealization ['@type'] = [];
 trainingDatasetInformationRealization ['prov:hadMember'] = [];
 trainingDatasetInformationRealization ['prov:hadMember'].append(trainingDatasetInfoElement)
 trainingDatasetInfoElement ['prov:wasMemberOf'] = trainingDatasetInformationRealization ['@id'];
 trainingDatasetInformationRealization ['prov:wasGeneratedBy'] = [];
 trainingDatasetInformationRealization ['prov:wasGeneratedBy'].append(dataTransformActivity)
 trainingDatasetInformationRealization ['@type'].append (context['owl:NamedIndividual']);
 trainingDatasetInformationRealization ['@type'].append (context['sao:InformationRealization']);
 trainingDatasetInformationRealization ['rdfs:comment'] = "Auto generated text: This information realization relates to training dataset.";

 global evalDatasetInfoElement
 evalDatasetInfoElement =createInformationElement ();
 evalDatasetInfoElement ['@type'].append(context['mls:Dataset']);
 evalDatasetInfoElement ['@type'].append (context['rains:RealizedObject']);
 evalDatasetInfoElement ['@type'].append (context['rains:RealizableObject']);
 evalDatasetInfoElement ['rains:hasRealizableObjectCharacteristic'] = [];
 

 global evalDatasetInformationRealization
 evalDatasetInformationRealization ={}
 evalDatasetInformationRealization ['@id'] = base_iri+ str(uuid.uuid4());
 evalDatasetInformationRealization ['@type'] = [];
 evalDatasetInformationRealization ['prov:hadMember'] = [];
 evalDatasetInfoElement ['prov:wasMemberOf'] = evalDatasetInformationRealization['@id'];
 evalDatasetInformationRealization ['prov:hadMember'].append(evalDatasetInfoElement)
 evalDatasetInformationRealization ['prov:wasGeneratedBy'] = [];
 evalDatasetInformationRealization ['prov:wasGeneratedBy'].append(dataTransformActivity)
 evalDatasetInformationRealization ['@type'].append (context['owl:NamedIndividual']);
 evalDatasetInformationRealization ['@type'].append (context['sao:InformationRealization']);
 evalDatasetInformationRealization ['rdfs:comment'] = "Auto generated text: This information realization relates to evaluation dataset.";

 global modelCreationActivity
 modelCreationActivity = {}
 modelCreationActivity ['@id'] = base_iri+ str(uuid.uuid4());
 modelCreationActivity ['@type'] = [];
 modelCreationActivity ['@type'].append(context['ep-plan:MultiActivity']);
 modelCreationActivity ['@type'].append (context['owl:NamedIndividual']);
 modelCreationActivity ['prov:used'] = [];
 modelCreationActivity ['prov:used'].append (trainingDatasetInformationRealization)
 modelCreationActivity ['prov:wasAssociatedWith'] = [];

 global model 
 model = {}
 model ['@id'] = base_iri+ str(uuid.uuid4());
 model ['@type'] = [];
 model ['prov:hadMember'] = [];
 model ['prov:wasGeneratedBy'] = [];
 model ['prov:wasGeneratedBy'].append(modelCreationActivity)
 model ['@type'].append (context['owl:NamedIndividual']);
 model ['@type'].append (context['sao:InformationRealization']);
 model ['rdfs:comment'] = "Auto generated text: This information realization relates to model implementation.";
 model ['prov:wasDerivedFrom'] = trainingDatasetInformationRealization ['@id'];

 global modelElement
 modelElement =createInformationElement ();
 modelElement ['@type'].append(context['mls:Model']);
 modelElement ['@type'].append (context['rains:RealizableObject']);
 modelElement ['@type'].append (context['rains:RealizedObject']);
 modelElement ['prov:wasMemberOf'] = model ['@id']

 #link to model implementation description collection  
 model ['prov:hadMember'].append (modelElement)
 
 global modelEvalActivity
 modelEvalActivity = {}
 modelEvalActivity ['@id'] = base_iri+ str(uuid.uuid4());
 modelEvalActivity ['@type'] = [];
 modelEvalActivity ['@type'].append(context['ep-plan:MultiActivity']);
 modelEvalActivity ['@type'].append (context['owl:NamedIndividual']);
 modelEvalActivity ['prov:used'] = [];
 modelEvalActivity ['prov:used'].append (model)
 modelEvalActivity ['prov:used'].append (evalDatasetInformationRealization)
 modelEvalActivity ['prov:wasAssociatedWith'] = [];

 global modelEvalDescription
 modelEvalDescription = {}
 modelEvalDescription ['@id'] = base_iri+ str(uuid.uuid4());
 modelEvalDescription ['@type'] = [];
 modelEvalDescription ['@type'].append (context['owl:NamedIndividual']);
 modelEvalDescription ['@type'].append (context['sao:InformationRealization']);
 modelEvalDescription ['prov:hadMember'] = [];
 modelEvalDescription ['prov:wasGeneratedBy'] = [];
 modelEvalDescription ['prov:wasGeneratedBy'].append(modelEvalActivity)
 modelEvalDescription ['rdfs:comment'] = "Auto generated text: This information realization relates to model evaluation.";
 modelEvalDescription ['prov:wasDerivedFrom'] = []
 modelEvalDescription ['prov:wasDerivedFrom'].append(model ['@id']);
 modelEvalDescription ['prov:wasDerivedFrom'].append(evalDatasetInformationRealization ['@id']);


#download the ttl file that can be imported to accountability fabric
def downloadAccountabilityTrace (g):
  ttlGraph = g.serialize(format="turtle")
  with open('graph.ttl', 'w') as writefile:
    writefile.write(ttlGraph)
    files.download('graph.ttl') 

def validateSHACLCosntraints (g,BASE_URL,SYSTEM_IRI):

 #take the generated accountability trace, combine with full plan taht contains SHACL constraints and evalaute
 validationGraph = copy.deepcopy(g)
 with urllib.request.urlopen(f"{BASE_URL}/getPlanElementsAsGraph?systemIri={SYSTEM_IRI}&stage=implementation") as url:
  planData = url.read().decode()
  validationGraph.parse(data=planData, format='turtle')
  #print(validationGraph.serialize(format="turtle").decode("utf-8"))

 r = validate(validationGraph, shacl_graph=None, ont_graph=None, inference=None, abort_on_error=False, meta_shacl=False,advanced=True, js=False,debug=False)
 conforms, results_graph, results_text = r
 
 #combine the validation report graph and combine with the plan (so we can retrieve the details of steps and constraints for the message given to the user) and query for failed constraints
 #TO DO we should probably only look at the constriants that are associated with steps for which model card can produce information for 
 graph = Graph()
 graph = results_graph + validationGraph
 qres = graph.query(
    """prefix prov: <http://www.w3.org/ns/prov#> 
       prefix rains: <https://w3id.org/rains#> 
       prefix sh: <http://www.w3.org/ns/shacl#> 
       prefix xsd: <http://www.w3.org/2001/XMLSchema#> 
       prefix ep-plan: <https://w3id.org/ep-plan#>
       SELECT DISTINCT ?activity ?stepLabel ?stepComment ?constraintLabel	?constraintComment ?message ?constraint
         WHERE {				
         ?validationReport sh:result ?resultBlankNode. 
         ?resultBlankNode sh:sourceShape ?constraintImplID; sh:resultMessage ?message. 
         ?constraint ep-plan:hasConstraintImplementation ?constraintImplID.
         ?constraint rdfs:label ?constraintLabel; rdfs:comment ?constraintComment; ep-plan:constrains ?step.
         ?step rdfs:label ?stepLabel;  rdfs:comment ?stepComment.
         ?activity ep-plan:correspondsToStep ?step. 
        }""")

 resultArray = []
 violatedConstraints = []
 for row in qres:
    entry = {}
    entry['activityURI'] = row[0]
    entry['stepLabel'] = row[1]
    entry['stepComment'] = row[2]
    entry['constraintLabel'] = row[3]
    entry['constraintComment'] = row[4]
    entry['message'] = row[5]
    entry['constraint'] = row[6]
    violatedConstraints.append(row[6])
    resultArray.append(entry)
    print(f"Not sufficient information provided for event { entry['stepLabel'] } ({ entry['stepComment']}). \n \t \x1b[31mFAILED:\x1b[0m Constraint  {  entry['constraintLabel']} ({entry['constraintComment']}) failed with the following message: \n \t \t {entry['message']}")

 #update the graph with records of which constraints have failed
 #TO DO consider checking and updating ep-plan relationships if the constraint has already been recorded as satisfied and now it failed  
 if len(violatedConstraints)>0:
  print("You can address these issues by adding additional elements in your Model Card or use the manual input interface of the Accountability Fabric")
  for resultEntry in resultArray:
   activity = URIRef(resultEntry['activityURI'])
   violated = URIRef("https://w3id.org/ep-plan#violated")
   constraint = URIRef(resultEntry['constraint'])
   g.add ((activity,violated,constraint))
 else: 
  print("No violations of any constraints were detected") 

 #check all constraints that were present and update them as satisfied 
 #TO DO consider checking and updating ep-plan relationships if the constraint has already been recorded as violated and now it wa ssatisfied
 qres = graph.query(
    """prefix prov: <http://www.w3.org/ns/prov#> 
       prefix rains: <https://w3id.org/rains#> 
       prefix sh: <http://www.w3.org/ns/shacl#> 
       prefix xsd: <http://www.w3.org/2001/XMLSchema#> 
       prefix ep-plan: <https://w3id.org/ep-plan#>
       SELECT DISTINCT ?activity ?constraint
         WHERE {				
          
         ?constraint ep-plan:hasConstraintImplementation ?constraintImplID.
         ?constraintImplID a sh:NodeShape.
         ?constraint ep-plan:constrains ?step.
         ?activity ep-plan:correspondsToStep ?step. 
        }""")

 for row in qres:
  if row[1] not in violatedConstraints:
   activity = URIRef(row[0])
   satisfied = URIRef("https://w3id.org/ep-plan#satisfied")
   constraint = URIRef(row[1])
   g.add ((activity,satisfied,constraint))







def createSingleManualMappingTask (jsonEl,conceptType):
  task={}
  task['w'] = 'Can\'t create a mapping task. There is no mapping defined for the entered conceptType value'
  task['jsonEl'] = jsonEl

  #model_card.model_details.owners 
  #TO DO 
  #if conceptType =='contact':
  # task['w'] = widgets.ToggleButtons(options=[('Include in agent description', 'agent'),  ('Do not map', 'none')],description='Select:',)
  # task['w'].style.button_width='200px'
  # task['title'] = 'Accountability Fabric maps owners as sao:AccountableActor responsible for the implementation task and linked to the model description via sao:isAccountableFor, however, it does, not specify further structure for contact details. Select action for the following '+conceptType+ ' :'
  # task['jsonElementText'] = jsonEl
  # task['conceptType'] = conceptType
   

  if conceptType =='use case':
   task['w'] = widgets.ToggleButtons(options=[('Intended Use Case', 'rains:IntendedUseCase'), (' Incorrect Use Case', 'rains:IncorrectUseCase'),  ('Do not map', 'none')],description='Select:',)
   task['w'].style.button_width='200px'
   task['title'] = 'Select type for the following '+conceptType+ ' :'
   task['jsonElementText'] = jsonEl
  
  if conceptType =='ethical considerations':
   task['w'] = widgets.ToggleButtons(options=[('Risk', 'rains:Risk'), ('Bias', 'rains:Bias'),('Include in model overview', 'model overview'), ('Do not map', 'none')],description='Select:',)
   task['w'].style.button_width='200px'
   task['title'] = 'Select type for the following '+conceptType+ ' :'
   task['jsonElementText'] = jsonEl["name"]
   task['conceptType'] = conceptType
   try:
    task['mitigationStrategy'] = jsonEl["mitigation_strategy"]
   except KeyError:
    pass

  #if conceptType =='tradeoffs': 
   #task['w'] = widgets.ToggleButtons(options=[('Limitation', 'rains:Limitation'),('Risk', 'rains:Risk'), ('Bias', 'rains:Bias'), ('Include in model overview', 'model overview'), ('Do not map', 'none')],description='Select:',)
   #task['w'].style.button_width='200px'
   #task['title'] = 'Select type for the following '+conceptType+ ' :'
   #task['jsonElementText'] = jsonEl
   #task['conceptType'] = conceptType
  
  if conceptType =='model architecture': 
   task['w'] = widgets.ToggleButtons(options=[('Algorithm', 'mls:Algorithm'), ('Include in model overview', 'model overview'), ('Do not map', 'none')],description='Select:',)
   task['w'].style.button_width='200px'
   task['title'] = 'Select type for the following '+conceptType+ ' :'
   task['jsonElementText'] = jsonEl
   task['conceptType'] = conceptType

  #if conceptType =='model input format': 
   #task['w'] = widgets.ToggleButtons(options=[ ('Include in model overview', 'model overview'), ('Do not map', 'none')],description='Select:',)
   #task['w'].style.button_width='200px'
   #task['title'] = 'Select type for the following '+conceptType+ ' :'
   #task['jsonElementText'] = jsonEl
   #task['conceptType'] = conceptType

  #if conceptType =='model output format': 
   #task['w'] = widgets.ToggleButtons(options=[ ('Include in model overview', 'model overview'), ('Do not map', 'none')],description='Select:',)
   #task['w'].style.button_width='200px'
   #task['title'] = 'Select type for the following '+conceptType+ ' :'
   #task['jsonElementText'] = jsonEl
   #task['conceptType'] = conceptType
   
  return task

def createManualMappingTasks (json_object):
  tasks= []
  #add use cases if info present in json file
  try:
   for x in json_object ["considerations"]["use_cases"]: 
    task = createSingleManualMappingTask (x,'use case')
    tasks.append (task)
  except KeyError:
    pass
 
  #try:
  # for x in json_object ["model_details"]["owners"]: 
  #  task = createSingleManualMappingTask (x,'contact')
  #  tasks.append (task)
  #except KeyError:
  #  pass

  try:
   for x in json_object ["considerations"]["ethical_considerations"]: 
    task = createSingleManualMappingTask (x,'ethical considerations')
    tasks.append (task)
  except KeyError:
    pass

  #try:
   #for x in json_object ["considerations"]["tradeoffs"]: 
    #task = createSingleManualMappingTask (x,'tradeoffs')
    #tasks.append (task)
  #except KeyError:
    #pass 

  try: 
    x = json_object ["model_parameters"]["model_architecture"]
    task = createSingleManualMappingTask (x,'model architecture')
    tasks.append (task)
  except KeyError:
    pass

  #try: 
    #x = json_object ["model_parameters"]["input_format"]
    #task = createSingleManualMappingTask (x,'model input format')
    #tasks.append (task)
  #except KeyError:
    #pass
  
  #try: 
    #x = json_object ["model_parameters"]["output_format"]
    #task = createSingleManualMappingTask (x,'model output format')
    #tasks.append (task)
  #except KeyError:
    #pass

  if len(tasks) > 0: 
    print('The following information elements could not be mapped automatically. Please select the appropriate options manualy\n')
  for task in tasks: 
    print('\n')
    print(task['title'])
    print(task['jsonElementText'])  
    try:
      print('\n!The following mitigation strategy will be also mapped:' + task['mitigationStrategy'])
    except KeyError:
      pass  
    displayWidget(task['w'])
  return tasks




def mapModelMetaData (json_object):
  
  #map training dataset
  try:
    trainingDatasetInfoElement ['rdfs:label'] = json_object ["model_parameters"]["data"]["train"]["name"]
  except KeyError:
    pass
  try:
    trainingDatasetInfoElement ['rdfs:comment'] = json_object ["model_parameters"]["data"]["train"]["link"]
  except KeyError:
    pass

  try:
   for x in json_object ["model_parameters"]["data"]["train"]["graphics"]["collection"]: 
    try:  
     el = createInformationElement ()
     el ['@type'].append (context['mls:DatasetCharacteristic']);
     try:
      el['rdfs:label'] = x ["name"]
     except KeyError:
      pass
     try:
      el['rains:hasBase64Image'] = x ["image"]
     except KeyError:
      pass
     el ['prov:wasMemberOf'] = trainingDatasetInformationRealization ['@id']
     trainingDatasetInformationRealization['prov:hadMember'].append(el)
     trainingDatasetInfoElement ['rains:hasRealizableObjectCharacteristic'].append(el)
    except KeyError:
     pass
  except KeyError:
     pass
  
  #map eval dataset
  try:
    evalDatasetInfoElement ['rdfs:label'] = json_object ["model_parameters"]["data"]["eval"]["name"]
  except KeyError:
    pass
  try:
    evalDatasetInfoElement ['rdfs:comment'] = json_object ["model_parameters"]["data"]["eval"]["link"]
  except KeyError:
    pass

  try:
   for x in json_object ["model_parameters"]["data"]["eval"]["graphics"]["collection"]: 
    try:  
      el = createInformationElement ()
      el ['@type'].append (context['mls:DatasetCharacteristic']);
      try:
       el['rdfs:label'] = x ["name"]
      except KeyError:
       pass
      try:
       el['rains:hasBase64Image'] = x ["image"]
      except KeyError:
       pass
      el ['prov:wasMemberOf'] = evalDatasetInformationRealization ['@id']
      evalDatasetInformationRealization['prov:hadMember'].append(el)
      evalDatasetInfoElement ['rains:hasRealizableObjectCharacteristic'].append(el)
    except KeyError:
     pass
  except KeyError:
     pass   
  
  #create accountable agents
  try:
   for x in json_object ["model_details"]["owners"]: 
    accountableAgent ['@type'] = [];
    accountableAgent ['@type'].append (context['owl:NamedIndividual']);
    accountableAgent ['@type'].append (context['sao:AccountableAgent']);
    accountableAgent ['rdfs:label']  = x['name']
    accountableAgent ['rdfs:comment']  = x['contact']
    dataTransformActivity ['prov:wasAssociatedWith'].append(accountableAgent)
    modelEvalActivity['prov:wasAssociatedWith'].append(accountableAgent)
    modelCreationActivity['prov:wasAssociatedWith'].append(accountableAgent)
    
    #also record teh owners as information elements in the information relaization corresponding to model component
    model['prov:hadMember'].append(accountableAgent)
    accountableAgent ['prov:wasMemberOf'] = model ['@id']
    accountableAgent ['sao:isAccountableFor'] = modelElement['@id']
  except KeyError:
    pass

  #map model overview
  try:
   modelElement ['rdfs:comment'] = json_object["model_details"].pop('overview')
  except KeyError:
    pass 
  #map model name
  try: 
   modelElement ['rdfs:label'] = json_object["model_details"].pop('name')
  except KeyError:
    pass
  #map model license
  try: 
   el = createInformationElement ()
   el ['@type'].append (context['dc:LicenseDocument']); 
   el['rdfs:comment']  = json_object["model_details"].pop('license')
   el['rdfs:label']  = "License (default label)"
   el ['prov:wasMemberOf'] = model ['@id']
   model['prov:hadMember'].append(el)
  except KeyError:
    pass
  #map references
  try:
    modelElement['rdfs:seeAlso'] = json_object["model_details"].pop('references')
  except KeyError:
    pass

  #map model Input format
  try:
    modelElement['rains:modelInputFormat'] = json_object["model_parameters"].pop('input_format')
  except KeyError:
    pass
  #map model Output format
  try:
    modelElement['rains:modelOutputFormat'] = json_object["model_parameters"].pop('output_format')
  except KeyError:
    pass   
  #map version
  try:
    modelElement['rains:version'] = json_object["model_details"]["version"].pop('name')
  except KeyError:
    pass
  #map version date
  try:
    modelElement['rains:versionDate'] = json_object["model_details"]["version"].pop('date')
  except KeyError:
    pass
   #map version diff
  try:
    modelElement['rains:versionNote'] = json_object["model_details"]["version"].pop('diff')
  except KeyError:
    pass
  
  #map model_card.quantitative_analysis.performance_metrics
  try: 
   for x in json_object ["quantitative_analysis"]["performance_metrics"]:
    try:  
      el = createInformationElement ()
      el ['@type'].append (context['rains:EvaluationMeasure']);
    
      el2 = createInformationElement ()
      el2 ['@type'].append (context['rains:EvaluationResult']);
      el2 ['@type'].append (context['mls:ModelEvaluation']);
      el2 ['rains:isEvaluationResultOf']= el;
      el2['rdfs:label']  = "Evaluation Result (default label)"
      try:
       el['rdfs:label'] = x ["type"]
      except KeyError:
       pass
      try:
       el['rains:computedOnSlice'] = x ["slice"]
      except KeyError:
       pass
      try:
       el2['rains:computedOnDecisionThreshold'] = x ["threshold"]
      except KeyError:
       pass
      try:
       el2['rains:hasResultValue'] = x ["value"]
      except KeyError:
       pass
      try:
       el2['rains:hasResultLowerBound'] = x ["confidence_interval"]["lower_bound"]
      except KeyError:
       pass
      try:
       el2['rains:hasResultUpperBound'] = x ["confidence_interval"]["upper_bound"]
      except KeyError:
       pass
      el ['prov:wasMemberOf'] = modelEvalDescription ['@id']
      el2 ['prov:wasMemberOf'] = modelEvalDescription ['@id']
      modelEvalDescription['prov:hadMember'].append(el)
      modelEvalDescription['prov:hadMember'].append(el2)
    except KeyError:
     pass 
  except KeyError:
     pass 
   
  
  #map model_card.quantitative_analysis.graphics.collection
  try:
   for x in json_object ["quantitative_analysis"]["graphics"]["collection"]: 
    try:  
     el = createInformationElement ()
     el ['@type'].append (context['rains:EvaluationMeasure']);
    
     el2 = createInformationElement ()
     el2 ['@type'].append (context['rains:EvaluationResult']);
     el2 ['@type'].append (context['mls:ModelEvaluation']);
     el2 ['rains:isEvaluationResultOf']= el;
     el2['rdfs:label']  = "Evaluation Result (default label)"
     try:
      el['rdfs:label'] = x ["name"]
     except KeyError:
      pass
     try:
      el2['rains:hasBase64Image'] = x ["image"]
     except KeyError:
      pass
     el ['prov:wasMemberOf'] = modelEvalDescription['@id']
     el2 ['prov:wasMemberOf'] = modelEvalDescription['@id']
     modelEvalDescription['prov:hadMember'].append(el)
     modelEvalDescription['prov:hadMember'].append(el2)
    except KeyError:
     pass
  except KeyError:
    pass
   
  #map limitations
  try:    
   for x in json_object ["considerations"]["limitations"]:
     try:
      el = createInformationElement ()
      el ['@type'].append (context['rains:Limitation']);
      el ['@type'].append (context['owl:NamedIndividual']);
      el['rdfs:comment'] = x
      el['rdfs:label']  = "Limitation (default label)"
      el ['prov:wasMemberOf'] = model['@id']
      model['prov:hadMember'].append(el)
     except KeyError:
      pass
  except KeyError:
   pass

   #map tradeoffs
  try:    
   for x in json_object ["considerations"]["tradeoffs"]:
     try:
      el = createInformationElement ()
      el ['@type'].append (context['rains:Tradeoff']);
      el ['@type'].append (context['owl:NamedIndividual']);
      el['rdfs:comment'] = x
      el['rdfs:label']  = "Tradeoff (default label)"
      el ['prov:wasMemberOf'] = model['@id']
      model['prov:hadMember'].append(el)
     except KeyError:
      pass
  except KeyError:
   pass
 
    #map users
  try:
    for x in json_object ["considerations"]["users"]:
     el = createInformationElement ()
     el ['@type'].append (context['rains:IntendedUserGroup']);
     el['rdfs:comment'] = x
     el['rdfs:label']  = "Intended Users (default label)"
     el ['prov:wasMemberOf'] = model ['@id']
     model['prov:hadMember'].append(el)
  except KeyError:
    pass

def createMappings (json_payload):
  initializeMappingTool ()
  json_object = json.loads(json_payload)
  mapModelMetaData (json_object)
  tasks = createManualMappingTasks (json_object)
  return tasks

def update_task_results (tasks):
   for task in tasks: 
    if  task['w'].value == 'none':
      #print ('skipping')
      continue
    elif task['w'].value =='model overview':
      text = " " + task['conceptType'] +":"+ task['jsonElementText']
      modelElement ['rdfs:comment'] = modelElement ['rdfs:comment'] + text 
      try:
       text = " Mitigation Strategy:"+ task['mitigationStrategy'] 
       modelElement ['rdfs:comment'] = modelElement ['rdfs:comment'] + text
      except KeyError:
       pass
      continue
    else:
      el = {}
      el ['@id'] = base_iri+ str(uuid.uuid4());
      el ['@type'] = [];
      el ['@type'].append(context[task['w'].value]);
      el ['@type'].append (context['owl:NamedIndividual']);
      el ['rdfs:comment'] =task['jsonElementText']
      label = task['w'].value.split(":")
      el['rdfs:label']  = label[1]+"(default label)" 
      try:
       el ['rains:hasMitigationStrategy'] = task['mitigationStrategy']
      except KeyError:
       pass
      el ['prov:wasMemberOf'] = model ['@id']
      model ['prov:hadMember'].append (el)
    #print(task)
    #print(task['title'])  
    #print(task['w'].value)

def mapAccountabilityTraceToPlan (PLAN_URL):
 with urllib.request.urlopen(PLAN_URL) as url:
  planData = json.loads(url.read().decode())
  #find corresponding plan elements

  dataTranformStep = ''
  evalDatasetVar = ''
  trainingDatasetVar = ''
  modelCreationStep = ''
  modelVar = ''
  modelEvaluationStep = ''
  modelEvalVar = ''

  for res in planData:
   #TODO !simplified assumption here: we assume that if step produces evaluation or training datset it is our data transform activity that produces training and eval dataset
   try:
    if res['outputType'] == 'https://w3id.org/rains#EvaluationDataset':
     #print(res)
     dataTranformStep = res['element']
     evalDatasetVar = res['output']
   except KeyError:
    pass
  
   try:
    if res['outputType'] == 'https://w3id.org/rains#TrainingDataset':
     #print(res)
     dataTranformStep = res['element']
     trainingDatasetVar = res['output']
   except KeyError:
    pass
  
   #TODO !simplified assumption here: we assume that if step produces model component  it is our model creation step 
   try:
    if res['outputType'] == 'https://w3id.org/rains#ModelComponent':
     #print(res)
     modelCreationStep = res['element']
     modelVar = res['output']
   except KeyError:
    pass

   try:
    if res['outputType'] == 'https://w3id.org/rains#Evaluation':
     #print(res)
     modelEvaluationStep = res['element']
     modelEvalVar = res['output']
   except KeyError:
    pass
  
  if trainingDatasetVar !='':
   el = {}
   el['@id'] = trainingDatasetVar
   trainingDatasetInformationRealization ['ep-plan:correspondsToVariable'] = el

  if evalDatasetVar !='':
   el = {}
   el['@id'] = evalDatasetVar
   evalDatasetInformationRealization ['ep-plan:correspondsToVariable'] = el

  if modelVar !='':
   el = {}
   el['@id'] = modelVar
   model ['ep-plan:correspondsToVariable'] = el

  if dataTranformStep !='':
   el = {}
   el['@id'] = dataTranformStep
   dataTransformActivity ['ep-plan:correspondsToStep'] = el

  if modelCreationStep !='':
   el = {}
   el['@id'] = modelCreationStep
   modelCreationActivity ['ep-plan:correspondsToStep'] = el

  if modelEvalVar !='':
   el = {}
   el['@id'] = modelEvalVar
   modelEvalDescription ['ep-plan:correspondsToVariable'] = el

  if modelEvaluationStep !='': 
   el = {}
   el['@id'] = modelEvaluationStep
   modelEvalActivity ['ep-plan:correspondsToStep'] = el 

def createAccounatbilityTrace (BASE_URL,SYSTEM_IRI):
  PLAN_URL = f"{BASE_URL}/getPlanElementsForImplementationStage?systemIri={SYSTEM_IRI}"

  mapAccountabilityTraceToPlan (PLAN_URL)
  resultJSON={}
  resultJSON['@context'] = context
  resultJSON['@graph'] =[]
  resultJSON['@graph'].append(trainingDatasetInformationRealization)
  resultJSON['@graph'].append(evalDatasetInformationRealization)
  resultJSON['@graph'].append(trainingDatasetInfoElement)
  resultJSON['@graph'].append(evalDatasetInfoElement)
  resultJSON['@graph'].append(model)
  resultJSON['@graph'].append(dataTransformActivity)
  resultJSON['@graph'].append(modelEvalActivity)
  resultJSON['@graph'].append(modelCreationActivity)
  resultJSON['@graph'].append(modelEvalDescription)

  # the result is a Python dictionary:
  jsonld = json.dumps(resultJSON)
  #print (jsonld)

  g = rdflib.Graph()
  g.parse(data=jsonld, format='json-ld')
  
  validateSHACLCosntraints (g,BASE_URL,SYSTEM_IRI)
  return g
   
