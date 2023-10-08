
def inferencia_producto(lista):
    tokens = {"OTHER": 0, "1200": 1, "1300": 2, "1400": 3, "1600": 4, "1700": 5, "1800": 6, "1900": 7, "2100": 8, "2400": 9, "2405": 9, "2600": 10, "2800": 11, "2805": 11, "AR": 12}
    input_list = lista.copy()  # Create a copy of the input list
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_ids = preprocess(input_list, tokens).to(device)
    model = AutoModelForSequenceClassification.from_pretrained("VCNC/bert_2").to(device)
    output = model(input_ids)['logits']

    mask = torch.full_like(output, -40)  # Initialize the mask with -4p

    for idx, element in enumerate(input_list):
        if element in tokens:
            token_id = tokens[element]
            mask[:, token_id] = output[:, token_id]

    # Find the element with the highest probability among the remaining elements
    max_index = torch.argmax(mask)

    #Aca le asigno la label al ouput que saque para despues eliminarlo de la lista
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
        elif (element == '2805' or element == '2804') and max_label == '2800':#Aca tuve que reducir los 05 o 04 a 00
            if element in lista:
                lista.remove(element) #Aca,lo elimino del input
                output_list.append(element)#Aca lo agrego a la lista del output y quedan ordenados los productos en base a como el modelo largo el output
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
