import datetime
import pymssql
import pyodbc
import sqlalchemy
from sqlalchemy import create_engine
from transformers import AutoModelForSequenceClassification
import torch
import pandas as pd
import numpy as np
import pytz
import random
SERVER = '192.168.0.247\SOLUTIIONWEB'
DATABASE = 'Solutiion'
USERNAME = 'sa'
PASSWORD = 'M0r3n02800!'
USER = 'sa'

#La función preprocess una vez entrada una lista con la descripción de los productos logra preprocesar la entrada para BERT
def preprocess(input:list[str], tokens):
    keys = [i for i in tokens.keys()]
    processed = np.zeros(len(input), dtype=int)
    for i in range(len(input)):
        for k in keys:
            if k in input[i]:
                processed[i] = tokens[k]
                break
            else:
                processed[i] = tokens['OTHER']
    return torch.tensor(np.asarray([processed]))


def comparison(machine):
    conn = pymssql.connect(server=SERVER, database=DATABASE, user=USER, password=PASSWORD)
    cursor = conn.cursor()
    select = f'''
    WITH CTE_ExtractChars AS (
    SELECT
        CASE
            WHEN CHARINDEX('-A', Codigos) > 0 THEN SUBSTRING(Codigos, CHARINDEX('-A', Codigos) + 2, 2)
            WHEN CHARINDEX('-T', Codigos) > 0 THEN SUBSTRING(Codigos, CHARINDEX('-T', Codigos) + 2, 2)
            ELSE NULL
        END AS PROCESOAT,
        Codigos
    FROM cnc
    WHERE Maquina = '{machine}'
          AND Codigos <> '()'
),

CTE_Comparison AS (
    SELECT CargaMaq2.*, cnc.*
    FROM CargaMaq2
    INNER JOIN cnc ON LEFT(CargaMaq2.CG_PROD, 7) = LEFT(cnc.Codigos, 7)
    WHERE CargaMaq2.CG_CELDA = '{machine}'
          AND cnc.Maquina = '{machine}' AND TIEMPOS BETWEEN DATEADD(DAY, -2, GETDATE()) AND GETDATE()
)

SELECT CG_ORDF, CG_PROD, CANT_PIEZAS, CANT,CODIGOS
FROM (
    SELECT
        CTE_Comparison.CG_ORDF,
        SUBSTRING(CTE_Comparison.CG_PROD,1,7) AS CG_PROD,
        CTE_Comparison.CANT_PIEZAS,
        CTE_Comparison.CANT,
        CTE_Comparison.T_CICLO,
        SUBSTRING(CTE_Comparison.CODIGOS,1,7) AS CODIGOS,

        ROW_NUMBER() OVER (PARTITION BY CG_ORDF, CG_PROD, CANT_PIEZAS, CANT ORDER BY T_CICLO DESC) AS RowNum
    FROM CTE_Comparison
) RankedData
WHERE RowNum = 1;
    '''
    cursor.execute(select)
    clear_query = f"UPDATE CARGAMAQ2 SET ESTADO = 0 WHERE CG_CELDA = '{machine}'"
    cursor.execute(clear_query)
    
    df = pd.read_sql_query(select, conn)
    df =  df.loc[df.groupby('CG_ORDF')['CANT_PIEZAS'].idxmax()]
    filtered_df = df.groupby('CG_PROD').filter(lambda x: len(x) > 1)
    cant_acumulada = 0
    duplicate_df = pd.DataFrame(columns=filtered_df.columns)  # Create an empty DataFrame to store the result
    column_to_check = 'CG_PROD'

    # Create a boolean mask to select rows where the specified column is not duplicated
    mask = ~df.duplicated(subset=[column_to_check], keep=False)

    # Apply the mask to the DataFrame to filter out rows where the column is duplicated
    unique_df = df[mask]
    # Iterate through the filtered_df
    for index, row in filtered_df.iterrows():
        cant = row['CANT']
        cant_piezas = row['CANT_PIEZAS']

        # Check if adding the current 'CANT' to the accumulated sum is lower or equal to 'CANT_PIEZAS'
        if cant_acumulada + cant <= cant_piezas:
            # Append the row to the result DataFrame
            duplicate_df = pd.concat([duplicate_df, pd.DataFrame([row])], ignore_index=True)
            
            # Update the accumulated sum
            cant_acumulada += cant
        else:
            # Break the loop if the condition is no longer met
            break

# Print the resulting DataFrame
    print(duplicate_df)
    print(unique_df)
    return unique_df,duplicate_df
 
def close(ordf):
    try:
        conn = pymssql.connect(server=SERVER, database=DATABASE, user=USER, password=PASSWORD)
        cursor = conn.cursor()

        close_query = f"UPDATE CARGAMAQ2 SET ESTADO = 1 WHERE CG_ORDF = '{ordf}'"
        cursor.execute(close_query)
        conn.commit()  # Commit the transaction
    except Exception as e:
        print(f"Error while updating: {str(e)}")
    finally:
        conn.close()
        

def rest(ordf,rest_cant,machine):
    try:
        conn = pymssql.connect(server=SERVER, database=DATABASE, user=USER, password=PASSWORD)
        cursor = conn.cursor()
        cant_query = f"UPDATE CARGAMAQ2 SET CANT_ACTUALIZADA = CANT WHERE CG_CELDA = '{machine}'"
        rest_query = f"UPDATE CARGAMAQ2 SET CANT_ACTUALIZADA = {rest_cant} WHERE CG_ORDF = '{ordf}'"

        cursor.execute(rest_query)
        conn.commit()  # Commit the transaction
    except Exception as e:
        print(f"Error while updating: {str(e)}")
    finally:
        conn.close()
        
#def acum(cant,cant_piezas):

def update_unique(unique_df,machine):

    for index, row in unique_df.iterrows():
        ordf = row['CG_ORDF']
        cant_piezas = row['CANT_PIEZAS']
        cant = row['CANT']
        if cant <= cant_piezas:
           close(ordf)
           print('PIEZAS',cant_piezas)
           print('ORDF',ordf)
           
           print('CANT',cant)
        if cant > cant_piezas:
            rest_cant = cant - cant_piezas
            print('rest_cant',rest_cant)
            print('ORDF',ordf)
            rest(ordf,rest_cant,machine)

def update_duplicate(duplicate_df,machine):

    for index, row in duplicate_df.iterrows():
        ordf = row['CG_ORDF']
        cant_piezas = row['CANT_PIEZAS']
        cant = row['CANT']
        if cant <= cant_piezas:
           close(ordf)
           print('PIEZAS',cant_piezas)
           print('ORDF',ordf)
           
           print('CANT',cant)
        if cant > cant_piezas:
            rest_cant = cant - cant_piezas
            print('rest_cant',rest_cant)
            print('ORDF',ordf)
            rest(ordf,rest_cant,machine)    
            
            
def get_producto(machine):
################################### SQL CONNECTION #########################

    conn = pymssql.connect(server=SERVER, database=DATABASE, user=USER, password=PASSWORD)
    cursor = conn.cursor()
    sql_select =  f'''select DISTINCT
CASE
        WHEN PATINDEX('%[0-9]%', c.DES_PROD) > 0
            THEN SUBSTRING(c.DES_PROD, PATINDEX('%[0-9]%', c.DES_PROD), 4)
        ELSE
            NULL
    END AS DES_PROD,
P.FE_EMIT
from cargamaq2 c
join programa p on p.CG_ORDF = c.CG_ORDF

where c.cg_celda like '%{machine}%' and C.ESTADO = 0
ORDER BY FE_EMIT ASC'''

    tokens = {"OTHER": 0, "1200": 1, "1300": 2, "1400": 3, "1600": 4, "1700": 5, "1800": 6, "1900": 7, "2100": 8, "2400": 9, "2405": 9, "2600": 10, "2800": 11, "2805": 11, "AR": 12}
    cursor.execute(sql_select)
    product_list= cursor.fetchall()

    lista = []
    for row in product_list:
        if row[0] == '2805' or  row[0] == '2804':
            lista.append('2800')
        if row[0] == '1805' :
            lista.append('1800')
            

        else:
            if row[0] != None :
                if row[0] != '500' :
                    lista.append(row[0])
        if row[0] in tokens:
            lista.append(row[0])
        else:
            lista.append("OTHER")        
    # Commit the changes to the database
    conn.commit()

    # Close the connection
    conn.close()
    return lista


def inferencia_producto(lista):
    tokens = {"OTHER": 0, "1200": 1, "1300": 2, "1400": 3, "1600": 4, "1700": 5, "1800": 6, "1900": 7, "2100": 8, "2400": 9, "2405": 9, "2600": 10, "2800": 11, "2805": 11, "AR": 12}
    input_list = lista.copy()  # Create a copy of the input list
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_ids = preprocess(input_list, tokens).to(device)
    model = AutoModelForSequenceClassification.from_pretrained("VCNC/bert_2").to(device)
    output = model(input_ids)['logits']

    mask = torch.full_like(output, -40)  # Initialize the mask with -5

    for idx, element in enumerate(input_list):
        if element in tokens:
            token_id = tokens[element]
            mask[:, token_id] = output[:, token_id]

    # Find the element with the highest probability among the remaining elements
    max_index = torch.argmax(mask)

    # If you want to get the actual label associated with the max_index in your 'tokens' dictionary
    for label, index in tokens.items():
        if index == max_index:
            max_label = label
            break
        
    # Create a new list to store removed elements
    output_list = []
    removed_count = 0  # Counter to track the number of removed elements
    max_removal_limit = 15  # Maximum number of elements to remove

    for element in input_list:
        if element == max_label and removed_count < max_removal_limit:
            if element in lista:
                lista.remove(element)
                output_list.append(element)
                removed_count += 1
        elif (element == '2805' or element == '2804') and max_label == '2800':
            if element in lista:
                lista.remove(element)
                output_list.append(element)
        elif (element == '1805') and max_label == '1800':
            if element in lista:
                lista.remove(element)
                output_list.append(element)
                
    print("Max Index:", max_index)
    print("Max Label:", max_label)
    print(f'lista: {lista}')
    print(f'mask: {mask}')
    print(f'output: {output}')
    
    return output_list

def insert_producto(producto,machine):
    # SQL CONNECTION
    conn = pymssql.connect(server=SERVER, database=DATABASE, user=USER, password=PASSWORD)
    cursor = conn.cursor()
# Check if the producto is 'other' and set it to '9100' if true
    if producto == 'OTHER':
        producto = '9100'

    # SQL INSERT statement to insert a single element into the PROD_PLAN table
    sql_insert = f"INSERT INTO ML_{machine}(PIEZA) VALUES (%s)"

    # Execute the INSERT statement with the producto as a parameter
    cursor.execute(sql_insert, (producto,))
    
    # Commit the transaction
    conn.commit()

    # Close the connection
    conn.close()
    
    
def get_piezas(machine):
################################### SQL CONNECTION #########################
    conn = pymssql.connect(server=SERVER, database=DATABASE, user=USER, password=PASSWORD)
    cursor = conn.cursor()
    sql_select =  f'''
   SELECT
    CASE 
        WHEN C.DES_PROD LIKE '%TUERCA RETEN%' THEN 'TUERCA'
        WHEN C.DES_PROD LIKE '%CAPUCHON%' THEN 'CAPUCHON'
        WHEN C.DES_PROD LIKE '%CUERPO%' THEN 'CUERPO'
        WHEN C.DES_PROD LIKE '%BONETE%' THEN 'BONETE'
        WHEN C.DES_PROD LIKE '%ANILLO%' THEN 'ANILLO'
        WHEN C.DES_PROD LIKE '%DISCO%' THEN 'DISCO'
        WHEN C.DES_PROD LIKE '%CPO.TOB%' OR C.DES_PROD LIKE '%CPO TOB%' OR C.DES_PROD LIKE '%CUERPO TOBERA%' THEN 'CPO.TOB'
        WHEN C.DES_PROD LIKE '%CONTRATUERCA%' THEN 'CONTRATUERCA'
        WHEN C.DES_PROD LIKE '%TOBERA%' THEN 'TOBERA'
        WHEN C.DES_PROD LIKE '%BUJE%' THEN 'BUJE'
        WHEN C.DES_PROD LIKE '%GUIA%' OR C.DES_PROD LIKE '%GUÍA%' THEN 'GUIA'
        WHEN C.DES_PROD LIKE '%RETEN%' THEN 'RETEN'
        WHEN C.DES_PROD LIKE '%VASTAGO%' OR C.DES_PROD LIKE '%VAST%' THEN 'VASTAGO'
        WHEN C.DES_PROD LIKE '%TORN%' THEN 'TORN'
        WHEN C.DES_PROD LIKE '%PORTADISCO%' THEN 'PORTADISCO'
        WHEN C.DES_PROD LIKE '%TUERCA%' THEN 'TUERCA'
        WHEN C.DES_PROD LIKE '%CABEZA%' THEN 'CABEZA'
        WHEN C.DES_PROD LIKE '%APOYO%' THEN 'APOYO'
        WHEN C.DES_PROD LIKE '%CAJA%' THEN 'CAJA'
        WHEN C.DES_PROD LIKE '%OBTURADOR%' THEN 'OBTURADOR'
        ELSE 'OTHER'
    END AS DES_PROD_CATEGORY,

	C.CG_ORDF,
	P.FE_EMIT
FROM 
    cargamaq2 C
JOIN PROGRAMA P ON C.CG_ORDF = P.CG_ORDF
WHERE 
    C.CG_CELDA LIKE '%{machine}%' and estado = 0
ORDER BY 
    P.FE_EMIT ASC;
    '''
    cursor.execute(sql_select)
    product_list= cursor.fetchall()

    lista = []
    for row in product_list:
        lista.append(row[0])
    # Commit the changes to the database
    conn.commit()

    # Close the connection
    conn.close()
    return lista

# def remove(lista):
#     removed_count = 0  # Counter to track the number of removed elements
#     max_removal_limit = 15  # Maximum number of elements to remove

#     for element in input_list:
#         if element == max_label and removed_count < max_removal_limit:  # Check if we haven't reached the removal limit
#             if element in lista:  # Check if the element is still in the original list
#                 lista.remove(element)  # Remove the element from the original list
#                 output_list.append(element)  # Keep the rest of the elements
#                 removed_count += 1  # Increment the removed_count

def inferencia_piezas(lista):
    tokens = {
    "OTHER":0,
    "CAPUCHON": 1,
    "CUERPO": 2,
    "CPO": 2,
    "BONETE": 3,
    "ANILLO": 4,
    "DISCO": 5,
    "CPO.TOB": 6,
    "CPO TOB": 6,
    "CONTRATUERCA": 7,
    "TOBERA": 8,
    "BUJE": 5,
    "GUIA": 10,
    "GUÍA": 10,  # Add both forms of GUIA
    "RETEN": 11,
    "VASTAGO": 12,
    "VAST": 12,
    "TORN": 13,
    "PORTADISCO": 14,
    "TUERCA": 15,
    "CABEZA": 16,
    "APOYO": 17,
    "CAJA": 18,
    "OBTURADOR": 19,
}
    input_list = lista  # Create a copy of the input list

  # Create a copy of the input list
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_ids = preprocess(lista, tokens).to(device)
    
    model = AutoModelForSequenceClassification.from_pretrained("VCNC/bert_piezas").to(device)
    output = model(input_ids)['logits']
    output_list = []
    print(f'lista: {lista}')

    mask = torch.full_like(output, -40)  # Initialize the mask with -5

    for idx, element in enumerate(lista):
        if element in tokens:
            token_id = tokens[element]
            mask[:, token_id] = output[:, token_id]

    # Find the element with the highest probability among the remaining elements
    max_index = torch.argmax(mask)

    # If you want to get the actual label associated with the max_index in your 'tokens' dictionary
    for label, index in tokens.items():
        if index == max_index:
            max_label = label
            break
        
    # Create a new list to store removed elements

    removed_count = 0  # Counter to track the number of removed elements
    max_removal_limit = 15  # Maximum number of elements to remove

    for element in input_list:
        if element == max_label and removed_count < max_removal_limit:  # Check if we haven't reached the removal limit
            if element in lista:  # Check if the element is still in the original list
                lista.remove(element)  # Remove the element from the original list
                output_list.append(element)  # Keep the rest of the elements
                removed_count += 1  # Increment the removed_count
                
    print("Max Index:", max_index)
    print("Max Label:", max_label)
    print(f'mask: {mask}')
    print(f'output: {output}')
    

    return output_list

def delete_sql(machine):
    conn = pymssql.connect(server=SERVER, database=DATABASE, user=USER, password=PASSWORD)
    cursor = conn.cursor()
    
    sql_index = f"DBCC CHECKIDENT('ML_{machine}', RESEED, 0);"
    # SQL INSERT statement to insert a single element into the PROD_PLAN table
    sql_delete = f"DELETE FROM ML_{machine}"

    cursor.execute(sql_delete)
    # Execute the INSERT statement with the producto as a parameter

    # Commit the transaction
    conn.commit()

    # Close the connection
    conn.close()

def insert_sql(pieza,machine):
    # SQL CONNECTION
    conn = pymssql.connect(server=SERVER, database=DATABASE, user=USER, password=PASSWORD)
    cursor = conn.cursor()
    
    # SQL INSERT statement to insert a single element into the PROD_PLAN table
    sql_insert = f"INSERT INTO ML_{machine} (PIEZA) VALUES (%s)"


    # Execute the INSERT statement with the producto as a parameter
    cursor.execute(sql_insert, (pieza,))
    
    # Commit the transaction
    conn.commit()

    # Close the connection
    conn.close()


def generate_random_color():
    color_names = ['Red', 'Green', 'Blue', 'Yellow', 'Purple', 'Orange', 'Pink', 'Cyan', 'Magenta', 'Brown']

    return random.choice(color_names)

def get_producto_output(machine):
################################### SQL CONNECTION #########################

    conn = pymssql.connect(server=SERVER, database=DATABASE, user=USER, password=PASSWORD)
    cursor = conn.cursor()
    sql_select =  f'''
    WITH CTE AS (
    SELECT
        PR.ID,
        AVG(CONVERT(INT, CNC.T_CICLO)) AS MAX_T_CICLO,
        SUB.DES_PROD,
        SUB.CG_PROD,
        SUB.CANT_ACTUALIZADA,
		CNC.CODIGOS,
		SUB.CG_ORDF,
        DENSE_RANK() OVER (ORDER BY SUB.CG_PROD) AS NewID, -- Create a new ID based on CG_PROD
        ROW_NUMBER() OVER (ORDER BY PR.ID ASC) AS RowNum
    FROM ML_{machine} PR
    JOIN (
        SELECT DISTINCT
            SUBSTRING(CG_PROD, 1, 7) AS CG_PROD,
            CG_CELDA,
            DES_PROD,
            FE_ENTREGA,
            CANT_ACTUALIZADA,
			CG_ORDF,
            ESTADO,
            FECHA_PREVISTA_FABRICACION,
            SUBSTRING(DES_PROD, PATINDEX('%[0-9]%', DES_PROD), 4) AS DES_PROD_SUBSTRING
        FROM CARGAMAQ2
    ) AS SUB
    ON LEFT(SUB.DES_PROD_SUBSTRING, 4) = LEFT(PR.PIEZA, 4)
    LEFT JOIN cnc ON LEFT(SUB.CG_PROD, 7) = LEFT(cnc.Codigos, 7)

    WHERE SUB.CG_CELDA LIKE '%{machine}%' and SUB.ESTADO = 0
    GROUP BY SUB.CG_PROD, PR.PIEZA, SUB.DES_PROD, SUB.CANT_ACTUALIZADA, PR.ID,SUB.CG_ORDF,CNC.CODIGOS
)
-- Group and order by NewID
SELECT 
    ID,
     MAX(MAX_T_CICLO) AS Max_T_CICLO,
     DES_PROD,
     CG_PROD,
	 CG_ORDF,
     SUBSTRING(LEFT(DES_PROD, CHARINDEX(' ', DES_PROD + '.') - 1), 1, 4)  AS DES_PROD_SUBS,
     CONVERT(int,CANT_ACTUALIZADA) as CANT_ACTUALIZADA
FROM CTE
GROUP BY ID,Max_T_CICLO,DES_PROD,CG_PROD,CANT_ACTUALIZADA,CG_ORDF
ORDER BY ID;
'''

    cursor.execute(sql_select)

    df = pd.read_sql_query(sql_select, conn)
    df = df.groupby('CG_ORDF').head(1)
    df = df.reset_index(drop=True)
    df.insert(5, 'INICIO', np.full(len(df), np.nan))
    df.insert(6, 'FIN', np.full(len(df), np.nan))
    tokens = {
    "ANIL":32,
    "APOY":177,
    "BONE":57,
    "CABE":6,
    "CAJA":18,
    "CAPU":20,
    "CPO.":23,
    "CUER":82,
    "DISC":15,
    "GUIA":16,
    "OBTU":20,
    "PORT":22,
    "TOBE":24,
    "TORN":12,
    "TUER":19,
    "VAST":19,
}
    print(df)
    for index, row in df.iterrows():
        if pd.isna(row['Max_T_CICLO']):
            # Get the first 4 characters of DES_PROD_SUBS
            first_4_chars = row['DES_PROD_SUBS'][:4]
            
            # Check if there is a matching token in the dictionary
            if first_4_chars in tokens:
                df.at[index, 'Max_T_CICLO'] = tokens[first_4_chars]
    df = df.drop(labels=['DES_PROD_SUBS'], axis=1)
    conn.commit()
    # Close the connection
    conn.close()
    
    return df


def get_piezas_output(machine):
################################### SQL CONNECTION #########################

    conn = pymssql.connect(server=SERVER, database=DATABASE, user=USER, password=PASSWORD)
    cursor = conn.cursor()
    sql_select =  f'''
    WITH CTE AS (
    SELECT
        PR.ID,
        AVG(CONVERT(INT, CNC.T_CICLO)) AS MAX_T_CICLO,
        SUB.DES_PROD,
        SUB.CG_PROD,
        SUB.CANT_ACTUALIZADA,
		CNC.CODIGOS,
		SUB.CG_ORDF,
        DENSE_RANK() OVER (ORDER BY SUB.CG_PROD) AS NewID, -- Create a new ID based on CG_PROD
        ROW_NUMBER() OVER (ORDER BY PR.ID ASC) AS RowNum
    FROM ML_{machine} PR
    JOIN (
        SELECT 
			CASE 
				WHEN C.DES_PROD LIKE '%TUERCA RETEN%' THEN 'TUERCA'
				WHEN C.DES_PROD LIKE '%CAPUCHON%' THEN 'CAPUCHON'
				WHEN C.DES_PROD LIKE '%CUERPO%' THEN 'CUERPO'
				WHEN C.DES_PROD LIKE '%BONETE%' THEN 'BONETE'
				WHEN C.DES_PROD LIKE '%ANILLO%' THEN 'ANILLO'
				WHEN C.DES_PROD LIKE '%DISCO%' THEN 'DISCO'
				WHEN C.DES_PROD LIKE '%CPO.TOB%' OR C.DES_PROD LIKE '%CPO TOB%' OR C.DES_PROD LIKE '%CUERPO TOBERA%' THEN 'CPO.TOB'
				WHEN C.DES_PROD LIKE '%CONTRATUERCA%' THEN 'CONTRATUERCA'
				WHEN C.DES_PROD LIKE '%TOBERA%' THEN 'TOBERA'
				WHEN C.DES_PROD LIKE '%BUJE%' THEN 'BUJE'
				WHEN C.DES_PROD LIKE '%GUIA%' OR C.DES_PROD LIKE '%GUÍA%' THEN 'GUIA'
				WHEN C.DES_PROD LIKE '%RETEN%' THEN 'RETEN'
				WHEN C.DES_PROD LIKE '%VASTAGO%' OR C.DES_PROD LIKE '%VAST%' THEN 'VASTAGO'
				WHEN C.DES_PROD LIKE '%TORN%' THEN 'TORN'
				WHEN C.DES_PROD LIKE '%PORTADISCO%' THEN 'PORTADISCO'
				WHEN C.DES_PROD LIKE '%TUERCA%' THEN 'TUERCA'
				WHEN C.DES_PROD LIKE '%CABEZA%' THEN 'CABEZA'
				WHEN C.DES_PROD LIKE '%APOYO%' THEN 'APOYO'
				WHEN C.DES_PROD LIKE '%CAJA%' THEN 'CAJA'
				WHEN C.DES_PROD LIKE '%OBTURADOR%' THEN 'OBTURADOR'
			ELSE 'OTHER'
    END AS DES_PROD_CATEGORY,
            SUBSTRING(CG_PROD, 1, 7) AS CG_PROD,
            CG_CELDA,
            DES_PROD,
            FE_ENTREGA,
            CANT_ACTUALIZADA,
			CG_ORDF,
            ESTADO,
            FECHA_PREVISTA_FABRICACION,
            LEFT(DES_PROD, CHARINDEX(' ', DES_PROD + ' ') - 1) AS DES_PROD_SUBSTRING
        FROM CARGAMAQ2 C
    ) AS SUB
    ON DES_PROD_CATEGORY = PR.PIEZA
    LEFT JOIN cnc ON LEFT(SUB.CG_PROD, 7) = LEFT(cnc.Codigos, 7)

    WHERE SUB.CG_CELDA LIKE '%{machine}%' and SUB.ESTADO = 0
    GROUP BY SUB.CG_PROD, PR.PIEZA, SUB.DES_PROD, SUB.CANT_ACTUALIZADA, PR.ID,SUB.CG_ORDF,CNC.CODIGOS
)
-- Group and order by NewID
SELECT 
    NewID,
     MAX(MAX_T_CICLO) AS Max_T_CICLO,
     DES_PROD,
     CG_PROD,
	 CG_ORDF,
     SUBSTRING(LEFT(DES_PROD, CHARINDEX(' ', DES_PROD + '.') - 1), 1, 4)  AS DES_PROD_SUBS,
     CONVERT(int,CANT_ACTUALIZADA) as CANT_ACTUALIZADA
FROM CTE
GROUP BY NewID ,Max_T_CICLO,DES_PROD,CG_PROD,CANT_ACTUALIZADA,CG_ORDF
ORDER BY NewID;
'''

    
    cursor.execute(sql_select)

    df = pd.read_sql_query(sql_select, conn)
    df = df.groupby('CG_ORDF').head(1)
    df = df.reset_index(drop=True)
    df.insert(5, 'INICIO', np.full(len(df), np.nan))
    df.insert(6, 'FIN', np.full(len(df), np.nan))
    tokens = {
    "ANIL":32,
    "APOY":177,
    "BONE":57,
    "CABE":6,
    "CAJA":18,
    "CAPU":20,
    "CPO.":23,
    "CUER":82,
    "DISC":15,
    "GUIA":16,
    "OBTU":20,
    "PORT":22,
    "TOBE":24,
    "TORN":12,
    "TUER":19,
    "VAST":19,
    "EJE": 29,
    "ARAN": 25,
    "CONT":42
}
    for index, row in df.iterrows():
        if pd.isna(row['Max_T_CICLO']):
            # Get the first 4 characters of DES_PROD_SUBS
            first_4_chars = row['DES_PROD_SUBS'][:4]
            
            # Check if there is a matching token in the dictionary
            if first_4_chars in tokens:
                df.at[index, 'Max_T_CICLO'] = tokens[first_4_chars]
    df = df.drop(labels=['DES_PROD_SUBS'], axis=1)
    conn.commit()
    # Close the connection
    conn.close()
    return df
from datetime import datetime, time


def change_time_to_7am(input_datetime):
    # Create a new time object with 7:00 AM
    new_time =time(7, 0, 0)

    next_day_time=input_datetime + datetime.timedelta(days=1)

    # Replace the time portion of the input datetime with 7:00 AM
    new_datetime = next_day_time.replace(hour=new_time.hour, minute=new_time.minute, second=new_time.second)
    
    return new_datetime



def end_time_iteration(start_datetime,end_datetime,cant_acumulada,t_ciclo,cant,cg):
    t_ciclo_mult=t_ciclo*cant_acumulada
    acum_code = 0


    end_datetime = start_datetime + datetime.timedelta(minutes=t_ciclo_mult)  # Update end_datetime here
    cant_acumulada += 1

    return cant_acumulada, end_datetime
#from datetime import datetime, timedelta

def generate_datetime_range(start_datetime ):

    current_date = start_datetime + timedelta(days=1)  # Move to Monday
    return current_date




def loops(start_datetime,start_datetime_given,cant,cant_acumulada,t_ciclo,cg,start_weekday):
    end_datetime = datetime.datetime(2023,8, 16, 10, 0, 0)
    start_datetime,end_time = set_end(start_weekday,start_datetime)
    
    while end_datetime.time() < end_time:
        
        cant_acumulada, end_datetime=end_time_iteration(start_datetime,end_datetime,cant_acumulada,t_ciclo,cant,cg)
        if cant != 0:
            cant=cant-1
            print(cant, end_datetime)

        else:
            return end_datetime, start_datetime,cant_acumulada, cant
    
    rest=  cant-cant_acumulada
    rest_min = rest * t_ciclo
    cant_acumulada=0
    print(f'cant:{cant}   {cg}')
    print(cant_acumulada,t_ciclo,t_ciclo*cant_acumulada,end_datetime)

    print('NO TERMINO TODAS LAS PIEZAS EN UN DIA')
    
    
    start_datetime=change_time_to_7am(start_datetime)

    while end_datetime.time() < end_time:
        cant_acumulada, end_datetime=end_time_iteration(start_datetime,end_datetime,cant_acumulada,t_ciclo,rest,cg)
        cant=cant-1

        print(cant, end_datetime)

        if cant == 0:
            print(f'{cant_acumulada} es igual a{rest}')

            return end_datetime, start_datetime,cant_acumulada,cant
    print(f'cant:{cant}   {cg}')
    print('llego al final')

    return end_datetime, start_datetime,cant_acumulada,cant


import calendar
import datetime
from datetime import timedelta
def calculate_end_datetime(df, start_datetime_given, cant, t_ciclo, end_time,start_time, cg,index):
    # Ini-datetime=tialize variables

    start_datetime=start_datetime_given
    end_datetime = datetime.datetime(2023,8, 16, 10, 0, 0)
    rest_min = 0
    rest=0
    cant_acumulada=0

    one_cycle_datetime=start_datetime + datetime.timedelta(minutes=t_ciclo)
    if one_cycle_datetime.time() > end_time:
        start_datetime=change_time_to_7am(start_datetime)
    print("WEEKDAY:",start_datetime.weekday())
    start_weekday =start_datetime.weekday()
    while cant > 0:
        end_datetime, start_datetime,cant_acumulada,cant =loops(start_datetime,start_datetime_given,cant,cant_acumulada,t_ciclo,cg,start_weekday)
        
    print("start:",start_datetime_given)
    print("rest:",rest)            
    print("cant:",cant)
    print("cant_acumulada:",cant_acumulada)
    print("next_dat_7_am:",start_datetime)

    print("rest_min:",rest_min)
    print("minutes:",cant_acumulada*t_ciclo)
    print("end:",end_datetime)
    print("cg:",cg)


    rest= cant-cant_acumulada + 1
    rest_min = rest * t_ciclo

        
    print('CACACACACACACACACACACACACACACAC',start_datetime_given )
    print('CACACACACACACACACACACACACACACAC',end_datetime )

    
    return end_datetime, start_datetime_given


def startend(df,start_date):
    print(df)
    # Assuming you have the 'end_time' defined elsewhere in your code, such as:
    end_time = datetime.time(16, 0)
    start_time = datetime.time(7, 0)


    df['Max_T_CICLO'] = df['Max_T_CICLO'].fillna(82)
    # Convert 'Max_T_CICLO' and 'CANT' columns to integers
    df['Max_T_CICLO'] = df['Max_T_CICLO'].astype(int)
    df['CANT_ACTUALIZADA'] = df['CANT_ACTUALIZADA'].astype(int)

    # # Initialize the 'start' and 'end' columns
    df['INICIO'] = start_date
    df['FIN'] = start_date
    # Calculate 'start' and 'end' for each row
    for index, row in df.iterrows():
        if index== 0:
            previous_end_time = start_date
        else:
            previous_end_time = df.at[index - 1, 'FIN']
        start_datetime=previous_end_time

        cant = row['CANT_ACTUALIZADA']
        t_ciclo = row['Max_T_CICLO'] 
        t_ciclo= t_ciclo
        cg = row['CG_PROD']
        minutes_to_add= t_ciclo * cant

        end_datetime, next_start_datetime = calculate_end_datetime(df,start_datetime,cant,t_ciclo,end_time,start_time,cg,index)

        df.at[index, 'INICIO'] = next_start_datetime
        df.at[index, 'FIN'] = end_datetime


    return next_start_datetime,end_datetime
 

def delete_planner(machine):
        # Create a connection to SQL Server using pymssql
    conn = pymssql.connect(server=SERVER, user=USERNAME, password=PASSWORD, database=DATABASE)
    cursor = conn.cursor()
    # Create a SQLAlchemy engine using the pymssql connection
    engine = sqlalchemy.create_engine(f"mssql+pymssql://{USERNAME}:{PASSWORD}@{SERVER}/{DATABASE}")
    
    delete = f'''
    DELETE FROM PLANNER_{machine}
    '''

    cursor.execute(delete)

    conn.close()

def insertdf(df,machine):
        # Create a connection to SQL Server using pymssql
    conn = pymssql.connect(server=SERVER, user=USERNAME, password=PASSWORD, database=DATABASE)
    cursor = conn.cursor()
    # Create a SQLAlchemy engine using the pymssql connection
    engine = sqlalchemy.create_engine(f"mssql+pymssql://{USERNAME}:{PASSWORD}@{SERVER}/{DATABASE}")
    
    # Assuming you have a DataFrame named 'df' and you want to write it to a table named 'table_name'
    df.to_sql(f"PLANNER_{machine}", engine, if_exists='replace', index=False)

    # Close the database connection

    conn.close()
    
import json
def config_json():
    try:
        with open('config.json', 'r') as config_file:
            config = json.load(config_file)
            
            # Convert end_time strings to datetime objects
            for machine_config in config.values():
                for day, end_time_str in machine_config.items():
                    if day.isdigit():
                        hour, minute = map(int, end_time_str.split(':'))
                        end_time_dt = time(hour, minute)
                        machine_config[day] = end_time_dt
            
            return config
    except Exception as e:
        print(f"Error loading config.json: {e}")
        return {}

def set_end(start_weekday, start_datetime):
    start_datetime = start_datetime
    end_time = datetime.time(16,0)
    config = config_json()
    for machine_name, machine_config in config.items():
        if machine_name == machine:
            for day, end_time_config in machine_config.items():
                if start_weekday == int(day):
                    end_time = end_time_config  # Update end_time if a match is found
                    print('SUNDAYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY')
    
    if start_datetime.weekday() == 6:
        start_datetime = generate_datetime_range(start_datetime)
    return start_datetime, end_time

        
def main():
    import datetime 

    global machine
    global start_date
    global start


    start = datetime.time(7, 0)

    
# Assign a specific start date and time to the first element
    machine = 'CN6'
    start_date = datetime.datetime(2023, 10, 6, 7, 0) # September 13, 2023, 7:00 AM
    unique_df,duplicate_df = comparison(machine)

    update_unique(unique_df,machine)
    update_duplicate(duplicate_df,machine)
    
    if machine == 'CN5' or machine == 'CM1':
        prod_inicial = get_producto(machine)
        print(prod_inicial)
        output_prod_accumulated = []
        delete_sql(machine)
        
        while prod_inicial:
             prod_final = inferencia_producto(prod_inicial)
             output_prod_accumulated.extend(prod_final)
    # Insert each element from lista_final into the PROD_PLAN table
             for producto in prod_final:
                 insert_sql(producto,machine)

        print("output_list:", output_prod_accumulated)
        df = get_producto_output(machine)
        print(df)
            
        startend(df,start_date)
        print(df)
        delete_planner(machine)
        insertdf(df,machine)
        
    else:
        lista_inicial = get_piezas(machine)
        print(lista_inicial)
        output_list_accumulated = []
        delete_sql(machine)
        
        while lista_inicial:
             lista_final = inferencia_piezas(lista_inicial)
             output_list_accumulated.extend(lista_final)
     # Insert each element from lista_final into the PROD_PLAN table
             for pieza in lista_final:
                insert_sql(pieza,machine)

        print("output_list:", output_list_accumulated)
        df = get_piezas_output(machine)
        print(df)
        startend(df,start_date)
        delete_planner(machine)
        insertdf(df,machine)

            
if __name__ == "__main__":
    main()
