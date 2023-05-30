import pandas as pd
import numpy as np
import warnings
import pandas as pd
import matplotlib.pyplot as plt 
import gradio as gr
import re
import tempfile
import os     
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import  FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns',500)

df_india = pd.read_excel(r'C:\Users\Gaurav Verma\1CIPL\ChatGpt_Fullsimulation_multipleCountry\data\india.xlsx',sheet_name='Sheet1')
df_india['SNO']=df_india['SNO'].str.lower()
matrix_india = pd.read_excel(r'C:\Users\Gaurav Verma\1CIPL\ChatGpt_Fullsimulation_multipleCountry\data\india.xlsx',sheet_name='Sheet2')
matrix_india.columns = matrix_india.columns.str.lower()

df_china = pd.read_excel(r'C:\Users\Gaurav Verma\1CIPL\ChatGpt_Fullsimulation_multipleCountry\data\china.xlsx',sheet_name='Sheet1')
df_china['SNO']=df_china['SNO'].str.lower()
matrix_china = pd.read_excel(r'C:\Users\Gaurav Verma\1CIPL\ChatGpt_Fullsimulation_multipleCountry\data\china.xlsx',sheet_name='Sheet2')
matrix_china.columns = matrix_china.columns.str.lower()

df_india['Country']='india'
df_china['Country']='china'

df_india.reset_index(inplace=True)
df_china.reset_index(inplace=True)

alldf=[df_india,df_china]
allmatrix=[matrix_india,matrix_china]

raw_df=pd.DataFrame()

for x in range(len(alldf)):
    df=alldf[x]
    matrix=allmatrix[x]
    unique_sku=df['SNO'].unique()
    for a in range(0,len(unique_sku)):
        index=a
    #     -0.2=20%
        for b in np.arange(-0.2, 0.21, 0.01):
            if (b ==0.0):
                continue
            else:
                b=round(b,4)
                empty_new_price = [0]*len(df['SNO'])
                empty_new_price[int(index)]=b
                ### real story starts
                df['Price Change']=empty_new_price
                df['New Price']=df['Base Price']+(df['Base Price']*df['Price Change'])
                df['vol_uni']=df['Volume UC']*((df['New Price']/df['Base Price'])**df['ELASTICITY'])
                df['DIRECT DISTRIBUTE']=((df['vol_uni']-df['Volume UC'])*(1+df['Walk Rate']))
                df['DIRECT DISTRIBUTE'] *= -1

                for i in df['SNO'].unique():
                    df['DIRECT CHANGE '+i]=df['DIRECT DISTRIBUTE']*matrix[i]
                list1=[]
                for i in df.filter(like='DIRECT CHANGE').columns:
                    c=df[i].sum()
                    list1.append(float(c))
                df['Switching Volume']    =list1
                df['Volume new']=np.where((df['Switching Volume'] + df['vol_uni']) < 0, 0, df['Switching Volume'] + df['vol_uni'])
                df['Value New']=df['Volume new']*df['New Price']
                df['Iter_sku']=unique_sku[a]
                df['Iter_percentage']=b
                df['Country']=df['Country']

                # raw_df=raw_df.append(df)
                # raw_df.append(df)
                raw_df = pd.concat([raw_df, df], ignore_index=True)


raw_df['Price Change']=round(raw_df['Price Change'],2)
# raw_df['Price Change']=round(raw_df['Price Change'],2)
raw_df['Iter_percentage']=raw_df['Iter_percentage']*100
raw_df['Iter_percentage'] = raw_df['Iter_percentage'].astype(float)
raw_df['Iter_percentage'] = raw_df['Iter_percentage'].astype(int)
raw_df['dfname'] = raw_df['Iter_percentage'].apply(lambda x: 'minus' + str(x)[1:] if x < 0 else 'plus' + str(x)[0:])
raw_df['dfname']=raw_df['dfname']+'_'+raw_df['Iter_sku'].astype(str)
raw_df['dfname']=raw_df['dfname']+'_'+raw_df['Country']
raw_df['dfname']=raw_df['dfname'].str.replace(' ','_')


unique_dfs = {}
for i in raw_df['dfname'].unique():
    temp = raw_df[raw_df['dfname']==i]
    unique_dfs[i]=temp
    
story_df=raw_df.copy()
story_df['Price Change']=story_df['Price Change']*100
story_df['Price Change']=round(story_df['Price Change'],2)

list_story=' in '+story_df['Country']+' for sku '+story_df['SNO'].astype(str)+' if we change price by '+story_df['Price Change'].astype(str)+'% then the df used is ('+story_df['dfname']+')'+' and if Category volume change or impact then function is (defcvol) and if Category value change or impact then function is (defcval) and if Category price change or impact then function is (defcpri)  and and if Brand Owner volume change or impact then function is (defbovol) if Brand Owner value change or impact then function is (defboval)  if Brand Owner price change or impact then function is (defbopri)  and if Brand volume change or impact then function is (defbvol) if Brand value change or impact then function is (defbval)  if Brand price change or impact then function is (defbpri)  and if Container material volume change or impact then function is (defcmvol)  if Container material value change or impact then function is (defcmval)  if Container material price change or impact then function is (defcmpri) and if Pack size volume change or impact then function is (defpsvol)  if Pack size value change or impact then function is (defpsval)   if Pack size price change or impact then function is (defpspri) ;'
list_story2=[]
for i in list_story:
    if('0.0%' in i):
        continue
    else:
        list_story2.append(i)
raw_text=''
for i in list_story2:
    raw_text+=i+';' 

    
def_dict={'defcvol':'CATEGORY%volume',
'defcval':'CATEGORY%value',
'defcpri':'CATEGORY%price',
'defbovol':'BRAND OWNER%volume',
'defboval':'BRAND OWNER%value',
'defbopri':'BRAND OWNER%price',
'defbvol':'BRAND%volume',
'defbval':'BRAND%value',
'defbpri':'BRAND%price',
'defcmvol':'CONTAINER MATERIAL / CONTAINER TYPE / REFILLABLE%volume',
'defcmval':'CONTAINER MATERIAL / CONTAINER TYPE / REFILLABLE%value',
'defcmpri':'CONTAINER MATERIAL / CONTAINER TYPE / REFILLABLE%price',
'defpsvol':'PACK SIZE / UNITS PER PACKAGE%volume',
'defpsval':'PACK SIZE / UNITS PER PACKAGE%value',
'defpspri':'PACK SIZE / UNITS PER PACKAGE%price'}


os.environ["OPENAI_API_KEY"] ="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"
raw_text=raw_text.replace("\n","")

text_splitter = CharacterTextSplitter(
    separator=";",
    chunk_size=2000,
    chunk_overlap=50,
    length_function=len,
)
texts=text_splitter.split_text(raw_text) 

# download embedding from openai
embeddings = OpenAIEmbeddings()
docsearch = FAISS.from_texts(texts,embeddings)


#chain = load_qa_chain(OpenAI(temperature=1.0,chain_type="map_rerank"))
chain = load_qa_chain(OpenAI())



################################



def chatbot_response(country,Please_Ask_Question):

    country=country.lower()
    Please_Ask_Question='for this country '+country+' ' +Please_Ask_Question + '? then what is my df?, tell me one function for asked change or impacted variable ?'
    docs = docsearch.similarity_search(Please_Ask_Question)
    ans = chain.run(input_documents=docs, question=Please_Ask_Question)
    ans=ans.replace('(', '')
    ans=ans.replace(')', '')
    ans=ans.replace('.','')
    ans.lower()


    pattern_minus = r"\bminus\w+"
    pattern_def = r"\bdef\w+"
    pattern_plus = r"\bplus\w+"

    matches_plus = re.findall(pattern_plus, ans.lower())
    matches_minus = re.findall(pattern_minus, ans.lower())
    matches_def = re.findall(pattern_def, ans.lower())

    try:
        dataFrameName = matches_plus[0]
        function = matches_def[0]
    except:
        dataFrameName = matches_minus[0]    
        function = matches_def[0]
    
    df_use=unique_dfs[dataFrameName]



    def plot_graph(df_use):

        column_name = def_dict[function].split('%')[0]
        attribute_name=def_dict[function].split('%')[1]
        grouped_df = df_use.groupby(column_name).sum()[['Volume UC', 'Volume new', 'Value Sales ', 'Value New', 'Base Price', 'New Price']]
        grouped_df.reset_index(inplace=True)

        # Set the 'BRAND OWNER' column as the index for easy plotting
        grouped_df = grouped_df.set_index(column_name)

        # Reset the index to use the 'BRAND OWNER' as a column
        grouped_df = grouped_df.reset_index()

        # Melt the DataFrame to transform it into a long format
        
        if (attribute_name=='price'):
            melted_df = pd.melt(grouped_df, id_vars=column_name, value_vars=['Base Price', 'New Price'], var_name='Price Type', value_name='Price')
        elif (attribute_name=='value'):
            melted_df = pd.melt(grouped_df, id_vars=column_name, value_vars=['Value Sales ', 'Value New'], var_name='Value Type', value_name='Value')
        else:
            melted_df = pd.melt(grouped_df, id_vars=column_name, value_vars=['Volume UC', 'Volume new'], var_name='Volume Type', value_name='Volume')

        # Create the bar plot using seaborn
        if (attribute_name=='price'):
            ax = sns.barplot(x=column_name, y='Price', hue='Price Type', data=melted_df)
        elif (attribute_name=='value'):
            ax = sns.barplot(x=column_name, y='Value', hue='Value Type', data=melted_df)
        else:
            ax = sns.barplot(x=column_name, y='Volume', hue='Volume Type', data=melted_df)

        # Set plot labels and title
        plt.xlabel(column_name)
        plt.ylabel(attribute_name)
        plt.title('Comparison of '+attribute_name.upper()+' OLD vs NEW'+ ' in ' +df_use['Country'].unique().tolist()[0])

        # Add the actual values on top of the bars
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 5), textcoords='offset points',fontsize=7)

        # Automatically adjust the figure size to fit the content
        plt.tight_layout()

        # Save the plot as a temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_img:
            plt.savefig(temp_img.name)
            img_path = temp_img.name

        plt.close() 

        return img_path
    
    img = plot_graph(df_use)



    return img

# iface = gr.Interface(fn=chatbot_response, inputs="text", outputs="image", title="ROE Chatbot", description="Ask your questions and get instant responses.", examples=[['in india what will be brand owner new price if i increase price by 7 percent on pepsi max 1500']])
iface = gr.Interface(fn=chatbot_response, 
                     inputs=[
                        gr.inputs.Textbox(lines=1,placeholder="Please enter country "),
                        gr.inputs.Textbox(lines=4,placeholder="Please enter question ")
], outputs="image",
 title="ROE Chatbot",
 description="Ask your questions and get instant responses.", 
 examples=[['india','what will be brand owner new volume if i increase price by 7 percent on pepsi max 1500'],
            ['china','what will be category value if i increase price by 7 percent on pepsi max 1500'],
            ['india','if i increase price by 17 percent on pepsi max 1500 then what will be impact on price']])
iface.launch(debug=True,share=True)
