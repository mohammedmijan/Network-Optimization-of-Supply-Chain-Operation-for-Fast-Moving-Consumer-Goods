#Modules 
import pandas as pd
import numpy as np
import time
from scipy.optimize import linprog
import warnings


from sklearn.mixture import (
    BayesianGaussianMixture, 
    GaussianMixture
)
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
import seaborn as sns

#Universal Variables
warnings.filterwarnings("ignore")
all_date = pd.read_csv("Data\\date.csv")
#Nodes
product_code_all = pd.read_csv("Data\\Nodes\\Nodes.csv")
product_code_all = list(np.array(product_code_all['Node'].values))
sub_group_product_code = pd.read_csv("Data\\Nodes\\Node Types (Product Group and Subgroup).csv")
dataset_of_nodes = pd.read_csv("Data\\Nodes\\Nodes Type (Plant & Storage).csv")#Number of connection between plant and storage
#Edges
dataset_of_number_of_edges_group = pd.read_csv("Data\\Edges\\Edges (Product Group).csv")
dataset_of_number_of_edges_sub_group = pd.read_csv("Data\\Edges\\Edges (Product Sub-Group).csv")
dataset_of_number_of_edges_plant = pd.read_csv("Data\\Edges\\Edges (Plant).csv")
dataset_of_number_of_edges_storage_location = pd.read_csv("Data\\Edges\\Edges (Storage Location).csv")
#Temporal Dataset
dataset_production = pd.read_csv("Data\\Temporal Data\\Unit\\Production .csv")
dataset_Delivery_To_distributor = pd.read_csv("Data\\Temporal Data\\Unit\\Delivery To distributor.csv")
dataset_Factory_Issue = pd.read_csv("Data\\Temporal Data\\Unit\\Factory Issue.csv")
dataset_Sales_Order = pd.read_csv("Data\\Temporal Data\\Unit\\Sales Order.csv")

#Json file for dataset description
dataset_description = pd.DataFrame({
    "numberOfproduct" : len(product_code_all), 
    "numberOfproductGroup" : len(sub_group_product_code['Group'].unique()),
    "numberOfproductSubGroup" : len(sub_group_product_code['Sub-Group'].unique()),
    "numberOfproductPlant" : len(dataset_of_nodes['Plant'].unique()),
    "numberOfproductStorageLocation" : len(dataset_of_nodes['Storage Location'].unique()),
    "countOfproduct" : len(all_date["Date"])*len(product_code_all),
    "countOfproductGroup" : len(dataset_of_number_of_edges_group['GroupCode']),
    "countOfproductSubGroup" : len(dataset_of_number_of_edges_sub_group['SubGroupCode']),
    "countOfproductplant" : len(dataset_of_number_of_edges_plant['Plant']),
    "countOfproductStorageLocation" : len(dataset_of_number_of_edges_storage_location['Storage Location'])
}, index=[0])

#Variables Controlling Functions of Supply Chain Networks information
def variable_identification(product_code:str, date:str):#Feature Extraction

    global dataset_of_number_of_edges_plant
    global dataset_of_number_of_edges_storage_location
    global dataset_of_nodes
    global dataset_production
    global dataset_Delivery_To_distributor
    global dataset_Factory_Issue
    global dataset_Sales_Order

    #Calculating Number of Edges
    number_of_edges_plant = len(dataset_of_number_of_edges_plant.loc[dataset_of_number_of_edges_plant['node1'] == product_code])
    number_of_edges_storage_location = len(dataset_of_number_of_edges_storage_location.loc[dataset_of_number_of_edges_storage_location['node1'] == product_code])
    #Calculating Number of plant and storage
    number_of_nodes = len(dataset_of_nodes.loc[dataset_of_nodes['Node']==product_code])
    #Calculating Production 
    production_amount_date_wise = dataset_production.loc[dataset_production['Date']==date][product_code]
    ##Calculating delivery to distributer
    Delivery_To_distributor_amount_date_wise = dataset_Delivery_To_distributor.loc[dataset_Delivery_To_distributor['Date']==date][product_code]
    ##Calculating Factory Issue
    Factory_Issue_amount_date_wise = dataset_Factory_Issue.loc[dataset_Factory_Issue['Date']==date][product_code]
    ##Calculating Sales Order
    Sales_Order_amount_date_wise = dataset_Sales_Order.loc[dataset_Sales_Order['Date']==date][product_code]
    

    
    return pd.DataFrame({
    "Date":date,#For Explanation
    'Product Code':product_code,#DES, MILP, ML
    "Number of Edges(Plant)":number_of_edges_plant,#For Explanation
    "Number of Edges(Storage)":number_of_edges_storage_location,#For Explanation 
    "Number of Nodes":number_of_nodes,#For Explanation
    "Production":production_amount_date_wise,#DES, MILP, ML
    "Delivery to Distributer":Delivery_To_distributor_amount_date_wise,#DES, MILP, ML
    "Factory Issue":Factory_Issue_amount_date_wise,#DES,MILP, ML
    "Sales Order":Sales_Order_amount_date_wise#DES, MILP, ML
    })


#-----------------------------------------------Discrete Event Simulation------------------------------ 
def des(all_date:pd.DataFrame):
    #local all_date
    pre_pro_da_stor = pd.DataFrame({})# Previous Production data Storage
    pre_simulated_time_value_stor = pd.DataFrame({})# Previous Simulated Time Storage after using Max Plus
    pre_simulated_so_time_value_stor = pd.DataFrame({})# Previous Simulated Sales Order Time Storage after using Max Plus
    accepted_after_des = None# Previous product 

        #-----------Inverse Transform Method for Exponential Inter-arrival Times--------
        #-----------MAX + Algebra for Selection-----------------------------------------
    def ITM_Max_plus(R:float, production_on_date:float, sales_order:float, value:float, so_time:float):
        #Variables confirmation
        print(f'Random Variable: {R}')
        r_of_pro = np.divide(production_on_date,24*3600)#On date production within 24 hours
        print(f"Rate of Production:{r_of_pro}")
        r_of_sales_order = np.divide(sales_order,24*3600)#Sales Order within 24 hours
        print(f"Rate of Sales Order:{r_of_sales_order}")

        #Model of Production Quantity in On date
        X_pqod = -np.divide(np.log(R), r_of_pro)# Simulated time of next event for (max, +) comparison 
        print(f"   Production Quantity in On date Simulated Event Timing:{X_pqod[0]}")
        #Model of Sales Order fill up of Company
        X_sofu = -np.divide(np.log(R), r_of_sales_order)# Simulated time of next event for (max, +) comparison 
        print(f"   Sales Order Simulated Event Timing:{X_sofu[0]}")

        #------------(Max, +) algebra using in maximum production identification---------
        print("(max +) algebra checking for not accepting the previous and present maximum one.")
        #One the simulation processes is accepted
        #Logic of selection of using (Max, +) algebra
        so_time = so_time+float(X_sofu[0])
        pqod_time = value+float(X_pqod[0])
        max_plus = max(pqod_time, so_time)#Max Plus Mathematical model
        print(f"\nMax Plus Value:{max_plus}")
        if max_plus == pqod_time: #If the network needs more time for production than sales order, it can meet the sales order
            print("Production Time Higher.\n")
            remark = True
        else: 
            print("Sales Order Time Higher.\n")
            remark = False
        return max_plus, remark, so_time
    
    #Initialize the functions of ITM-Max-Plus
    #Date Selection loop
    j = 0
    for date in all_date['Date'].values:
        j+=1
        print(f"Serial Number of day:{j}") 
        #Product code selection loop
        for i,product_code in enumerate(product_code_all, start=1):
            print(f"Analyzing Network Serial No:{i}")
            print(f"Operation of Production Product Code of Network: {product_code}")
            date_values = variable_identification(product_code=product_code, date=date)
            number_of_nodes = date_values["Number of Nodes"].values
            #Sales order meets function recognition : The sales order product having in total hand in industry
            sales_order = date_values['Sales Order'].values
            production_on_date = date_values['Production'].values

            try:
                p_d_pro_quan = pre_pro_da_stor[product_code].values
                pre_simulated_time_value = float(pre_simulated_time_value_stor[product_code].values)
                pre_simulated_so_time_value = float(pre_simulated_so_time_value_stor[product_code].values)
            except:
                p_d_pro_quan = 0#previous date production quantity
                pre_simulated_time_value = 0
                pre_simulated_so_time_value = 0

            #-----Selection of Random Number for ITM-----
            if sales_order != 0 and production_on_date != 0:# Non zero of sales and production are selected
                production_on_date = date_values['Production'].values
                del_to_dis = date_values['Delivery to Distributer'].values
                fac_issue = date_values['Factory Issue'].values
                
                da_pro_quan = del_to_dis+fac_issue+p_d_pro_quan#On this date production quantity in hand
                print(f"Total Product in hand:{da_pro_quan}")

                #Selection of Random Variables according to priority basis
                if sales_order >= da_pro_quan:#Higher Sales Order than products
                    need_production = sales_order - da_pro_quan #the actual production needed on this date 
                    if production_on_date >= need_production:
                        R = 0.9#Mostly Prioritise than other Networks
                        print("The production quantity prviously in hand and production amount meets the sales order. So, the network is mostly prioritise.")
                        p_d_pro_quan = production_on_date - need_production#Reserve the products for extra poduction
                        #product_selection = True
                        pre_pro_da_stor[product_code] = p_d_pro_quan
                    else: 
                        if need_production - production_on_date < 100:#A factory needs minimum 100 unit production facilty
                            print("The production quantity in hand needs less than 100 units to fulfill sales order. So, the network is less partially prioritise. Because of at least 99 units of products may have put in invetory.")
                            R = 0.5#Less Partialy Priority
                            #product_selection = True
                            #print(f'Product needs to be produce: {need_production-product_selection}')
                        else:
                            print("The production amount does not meets the sales order.")
                            R = 0.2#Less Priority
                            #product_selection = False
                else:#lower Sales order than products quantity
                    print("The production quantity meets the sales order using production amount in hand. But, inventory needs higher which is costly. So, the network is partially prioritise. Besides, it may have meet for next sales order.")
                    R = 0.7#Partial Priority
                    p_d_pro_quan = production_on_date + da_pro_quan - sales_order 
                    #product_selection = True
                    pre_pro_da_stor[product_code] = p_d_pro_quan
                
                #ITM and Max + function output
                value, remark, so_time = ITM_Max_plus(R, production_on_date, sales_order, value=pre_simulated_time_value, so_time=pre_simulated_so_time_value)

                #Selection of Networks
                if remark:
                    #As having any accepted Supply Chain Network
                    pre_simulated_time_value_stor[product_code] = value
                    pre_simulated_so_time_value_stor[product_code] = so_time
                    print("______Presenting Selected Supply Chain Network Simulation_______\n")
                    print(f"Number of Nodes: {number_of_nodes}")
                    print(f"Production:{production_on_date}")
                    print(f"Sales Order:{sales_order}")
                    print(f"ITM and Max Plus Value:{value}")
                    print(f"ITM and Max Plus Sales order Time:{so_time}")
                    print("\nAccepted!!!")
                    try: 
                        accepted_after_des = accepted_after_des.merge(date_values, how='outer')
                        print(f"Accepted Serial:{accepted_after_des.index[-1]}")
                        print("Merger completed!!!\n")
                        #time.sleep(1)
                    except:
                        accepted_after_des = date_values #Loading First Supply Chain Network
                else:
                    print("____Not Accepted Network_____\n")
                    #time.sleep(2)
            else:
                print("The production quantity and sales order is zero unit.")
        
    return accepted_after_des


#------------------------Mixed Integer Linear Programming using--------------------- 
def milp_optimization(data_a_des:pd.DataFrame, l:float):
    #Making a Matrix with the equation for the constraints
    print("\n \n \n \n -------------MILP Optimization Start---------------\n \n \n \n ")
    """
    Cost Maximization =  x_so - (x_p+x_d+x_fs)
    Constraints all greater than zero
    x_p - m*x_so >= 0
    x_d - n*x_so >= 0
    x_fs - p*x_so >= 0
    a_p*x_p + a_d*x_d + a_fs*x_fs - a_so*x_so >= 0
    x_p+x_d+x_fs - l*x_so = 0 
    x_p, x_d, x_fs, x_so >= 0
    """
    data_a_des_pos = np.array(data_a_des.iloc[:,4:7])
    data_a_des_neg = np.multiply(-1, data_a_des.iloc[:,7:8])
    data_a_des_pn = np.hstack([data_a_des_pos, data_a_des_neg])

    k = np.divide(0.25,0.65) #cost ratio of delivery and production cost
    H = np.divide(0.1, 0.65)#cost ratio of factory issue and production cost

    m = np.divide(l, 1+k+H)#Percentage of production cost w.r.t total cost
    n = k*m#Percentage of delivery cost w.r.t total cost
    p = H*m#Percentage of factory issue cost w.r.t total cost

    #Constraints of presentation matrix
    m_1 = [1, 0, 0, -m]
    m_2 = [0, 1, 0, -n]
    m_3 = [0, 0, 1, -p]
    m_5 = [1, 1, 1, -l]

    c = [1, 1, 1, -1]#cost function presentation as maximization use negative value

    b_1 = [0, 0, 0, 0]
    b_2 = [0]

    
    list_index_nember = []
    for index_number,m_4 in enumerate(data_a_des_pn):
        print(f"Data Processing for Optimization: {data_a_des.loc[index_number, 'Product Code']}")
        a = [m_1, m_2, m_3, m_4]
        result = linprog(c = c, A_ub=a, b_ub=b_1, A_eq=[m_5], b_eq=b_2, method='simplex')
        if result['success'] == True:
            print("----Optimized----")
            list_index_nember.append(index_number)
            #time.sleep(1)
            
        else:
            print("___________Not Optimized________")

    data_a_milp = data_a_des.loc[list_index_nember]

    #Sorting the optimized value for most SCN identified
    pc_l = []
    cv_l = []
    d_f_s = pd.DataFrame({})
    for pc in data_a_milp['Product Code'].unique():
        counter_value = data_a_milp.loc[data_a_milp['Product Code'] ==  pc]['Product Code'].count()
        pc_l.append(pc)
        print(f"Product Code: {pc}")
        print(f"Product Code counter: {counter_value}")
        cv_l.append(counter_value)
    d_f_s['Product Code'] = pc_l
    d_f_s['Count'] = cv_l
    d_f_s_d = d_f_s.sort_values(by=['Count'], ascending=False)

    return data_a_milp, d_f_s_d

#-------------------------Machine Learning Prediction analysis------------------------------
def ml_predictions(d_f_s_d:pd.DataFrame):
    def feature_extraction(data:pd.DataFrame):
        #____________Feature Extraction for GMM and BGMM____________
        so_pd_dd_fi_dataset = data.loc[:, ["Production","Delivery to Distributer", 
                                            "Factory Issue", "Sales Order"]]
        X = np.array(so_pd_dd_fi_dataset)
        return X

    def model_development(n_components:int, selected_model:str, data:pd.DataFrame):

        X =  feature_extraction(data=data)
        
        #Scoring the models
        def scoring(estimator, X):
            return  estimator.bic(X)

        #Selection of Model
        if selected_model == "GMM":
            estimator = GaussianMixture()
            param_grid={
                "n_components": [n_components], 
                "covariance_type": ["full"], 
                "init_params": ['random_from_data'],
            }

        if selected_model == "BGMM":
            estimator = BayesianGaussianMixture()
            param_grid={
                "n_components": [n_components], 
                "covariance_type": ["full"], 
                "weight_concentration_prior": [1/n_components],
                "weight_concentration_prior_type": ['dirichlet_distribution'],
                "init_params": ['random_from_data'],
            }

        #Model Creation
        model = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring= scoring
            )
        model.fit(X)
        return X, model

    #Visualization and Presentation of Cluster Data
    def presentation_analysis(estimator, model_name:str, data:pd.DataFrame, n_components:int):
        #Prediction Variable and Future selected cluster Data
        product_code = data['Product Code'].values[0]
        X = feature_extraction(data=data)
        X_predict = estimator.predict(X)
        data_after_cluster = {}
        c_n_list = []
        l_list = []
        c_d = []
        c_d_c = []
        m_data = []
        range_data = []
        v_name_list = []

        fig_name_list = ["Production", "Delivery to Distributer", "Factory Issue", "Sales Order",
                        "Production Cluster", "Delivery to Distributer Cluster", "Factory Issue Cluster", "Sales Order Cluster"]
        colors = sns.color_palette(n_colors=n_components)
        x_dates = np.array(data.loc[:, ["Date"] ]).reshape(-1)

        dimentions = X.shape[1]*2

        #Plot No 1 for Data Presentation 
        fig1 = plt.figure(num=2*dimentions,figsize=(15, 30), layout="constrained")
        axs1 = fig1.subplot_mosaic(mosaic=[["Production", "Delivery to Distributer" ],
                                        ["Factory Issue", "Sales Order" ]])
        fig1.suptitle(t=f"{product_code} Product Data Presentation")
        for  n in range(0, dimentions-4):
            fig_name = fig_name_list[n]
            color = colors[n-3]
            axs1[fig_name].set_title(f"{fig_name} Data.")
            axs1[fig_name].scatter(x_dates,X[:, n], s=50, color=color, alpha=1.0)
            axs1[fig_name].set_ylabel(fig_name)
            axs1[fig_name].set_xlabel("Date")
            axs1[fig_name].set_xticks(rotation=15,ticks=[i for i in range(0, len(x_dates), 50)],  labels=[x_dates[i] for i in range(0, len(x_dates), 50)])
        plt.show()

        #Plot No 2 for Cluster Presentation
        fig2 = plt.figure(num=2*dimentions,figsize=(15, 30), layout="constrained")
        axs2 = fig2.subplot_mosaic(mosaic=  [["Production Cluster","Delivery to Distributer Cluster" ],
                                        ["Factory Issue Cluster","Sales Order Cluster" ]])
        fig2.suptitle(t=f"{product_code} Product Cluster Data Presentation")
        for  n in range(4, dimentions):
            fig_name = fig_name_list[n]
            color = colors[n-3]
            axs2[fig_name].set_title(f"{fig_name} Model: {model_name} Data.")
            for i , color in enumerate(colors):
                if not np.any(X_predict == i):
                    continue

                Y = X[X_predict == i, n-4]
                length = len(Y)
                c_name = f"c{i+1}"
                c_n_list.append(c_name)
                c_d_c.append(str(x_dates[X_predict == i]))
                l_list.append(length)
                c_d.append(str(Y))
                m_data.append(max(Y))
                range_data.append(max(Y)-min(Y))
                v_name_list.append(fig_name)

                area = np.pi * length
                axs2[fig_name].scatter([f"{c_name}" for j in range(0, length)],Y, s=area, color=color, alpha=0.6)
                axs2[fig_name].set_xlabel(fig_name)
                axs2[fig_name].set_xticks(ticks=[l for l in range(0, n_components, 1)])            

            #Cluster Data preparation
            data_after_cluster['Variable Name'] = v_name_list
            data_after_cluster['Cluster Name'] = c_n_list
            data_after_cluster['Cluster Length'] = l_list
            data_after_cluster['Cluster Data'] = c_d
            data_after_cluster['Cluster Date'] = c_d_c
            data_after_cluster["Range Length"] = range_data
            """data_after_cluster['Maximum Lenth'] = max(l_list)"""
            data_after_cluster['Maximum Data'] = m_data
        plt.show() 
        
        return pd.DataFrame(data_after_cluster)

    # Save the cluster data as json file for Data Visualization and Further Use
    def cluster_data_save(data_a_cluster:pd.DataFrame, product_code:str):
        data_selected = data_a_cluster.loc[(data_a_cluster["Variable Name"] == "Production Cluster") | (data_a_cluster["Variable Name"] =="Sales Order Cluster")]
        sort_cluster = sorted(data_selected['Cluster Length'], reverse=True)[:10]
        data_selected = data_selected.loc[data_selected['Cluster Length'].values >= min(sort_cluster) ]#
        data_selected.to_json(f"Data\\cluster\\{product_code} Cluster.json", index=False, orient="records", indent=5)
        return f"\n Json file of {product_code} Completed!!!\n"

    #Data Preparation and ML Application with predict the data
    print("\n\n____________Machine Learning Using for Predictions___________")
    print("_______Unsupervised Learning________\n")
    selected_model = input(str(f"Selected a MIXTURE model GMM/BGMM:"))
    n_components = input(f"Mention cluster number {selected_model}:")
    n_components = int(n_components)

    #Maximum, MidRange and Minimum Counted Optimized Value Presentation 
    length_counter = len(d_f_s_d['Product Code'])
    mid_range = int(np.divide(length_counter, 2) + 1)
    visualization_data_number = [0, 1, 2, 3, 4]

    #Creating model for Optimized SCN with data preparation
    for i, dfsdpc in enumerate(d_f_s_d['Product Code'].values):
        data_values = pd.DataFrame({})
        for da in all_date['Date'].values:
            data_value = variable_identification(product_code=dfsdpc, date=da)
            try:
                data_values = data_values.merge(data_value, how='outer')
            except:
                data_values = data_value
        try:
            training_data_for_pc_string = f"Data\\train\\{dfsdpc} training data.csv"
            data_values.to_csv(training_data_for_pc_string)
            print(f"Train data of {dfsdpc} is ready.")
        except:
            print("____Not Done____")

        #Traing the model and predict data
        print(f"Training the model of {dfsdpc}")
        X, model = model_development(n_components=n_components, selected_model=selected_model, data=data_values)
        print("________ML Model Completed!!!____________\n")
        
        #Data Presentation
        print("\n__________Data Presentation of Data________")
        data_a_cluster = presentation_analysis(estimator=model, model_name=selected_model, data=data_values, n_components=n_components)
        print("Cluster of Product Supply Chain Netwroks Save in Json File.")
        print(cluster_data_save(data_a_cluster=data_a_cluster, product_code=dfsdpc))
        
        
def data_analysis(data:pd.DataFrame, title:str):
    total_networks = data['Product Code'].count()
    number_of_product_selected = len(data['Product Code'].unique())
    data_value_count_value = data['Product Code'].value_counts().values
    data_value_count_key = data['Product Code'].value_counts().keys()
    maximum_product_values = data_value_count_value[0]
    maximum_product_selection = data_value_count_key[0]
    minimum_product_values = data_value_count_value[-1]
    minimum_product_selection = data_value_count_key[-1]
    colors = sns.color_palette(n_colors=len(data['Product Code'].unique()))
    fig, ax = plt.subplots()
    ax.bar(x=data_value_count_key, height=data_value_count_value, color = colors)
    plt.xticks(ticks=data_value_count_key, rotation=90)
    plt.xlabel("Product Code")
    plt.ylabel("Count")
    ax.set_title(title)
    plt.show()

    af_data_description = pd.DataFrame(
        {
            "Total Networks": total_networks,
            "Production Networks Quantity": number_of_product_selected,
            "Maximum Product Networks":maximum_product_values, 
            "Maximum Product Code Network":maximum_product_selection, 
            "Minimum Product Networks":minimum_product_values, 
            "Minimum Product Code Network":minimum_product_selection, 
        }, index=[0]
    )
    af_data_description.to_json(f'Data\\Data Presentation\\{title}.csv',orient='records', indent=6)
    return f"{title} completed!!!"

##Initaiting all functions for sequential identification of Supply Chain Networks
def initialize():
    global dataset_description
    print("Start to create json file.")
    dataset_description.to_json("Data\\Data Presentation\\Data_Set_Description.json", orient='records', indent=6)
    print("\n Finished to Save Dataset Description.")
    output1 = str(input("Do you want to simulate?\n")).upper()
    time.sleep(2)
    output2 = str(input("Do you want to optimize and predict?\n")).upper()
    yes = "yes".upper()
    if output1 == yes:
        accepted_after_des_final = des(all_date=all_date)
        print(data_analysis(data=accepted_after_des_final, title="After Discrete Event Simulation Product Networks Counts"))
    if output2 == yes:
        output2_1 = float(input("How much you want to operating cost?\n"))
        l = output2_1#Operating Cost ratio
        if output1 != yes:
            print("!Old data use!")
            #time.sleep(3)
            accepted_after_des_final = pd.read_csv("after DES Supply Chain Network.csv")
        accepted_after_milp_final, d_f_s_d = milp_optimization(data_a_des=accepted_after_des_final, l=l)
        print(data_analysis(data=accepted_after_milp_final, title="After MILP Product Networks Counts"))
        ml_predictions(d_f_s_d=d_f_s_d)
    else:
        print("Thanks for being with us.")

    #print(accepted_after_des_final)                    
    try:
        if output1 == yes:
            accepted_after_des_final.to_csv("after DES Supply Chain Network.csv", index=False)#load in CSV file
        if output2 == yes:
            accepted_after_milp_final.to_csv("after MILP Supply Chain Network.csv", index=False)#load in CSV file
        return "Done!!!"
    except:
        return "---Not Accepted---"
print(initialize())


