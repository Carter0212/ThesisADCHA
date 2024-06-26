import numpy as np
import pickle
import csv

## Parmeter List of deploy (Start)
seed_code = 880212
NumberOfStation = 5 #Number of Base Sations
 # Number of Fixed Wirless Access
min_Value = 0
max_Value = 100
decimal = 2
## Parmeter List of deploy (End)

def Prodcting_coords(Number ,x_Max,x_Min,y_Max,y_Min,decimal):
    coordinate = []
    while len(coordinate) < Number:
        x_coords = np.around(np.random.uniform(x_Min,x_Max),decimals=decimal)
        y_coords = np.around(np.random.uniform(y_Min,y_Max),decimals=decimal)
        new_coords =(x_coords,y_coords)

        if new_coords not in coordinate:
            coordinate.append(new_coords)
    
    return np.array(coordinate)

if __name__ == '__main__':
    FWA = [10,20,30,40,50]
    for NumberOfFWA in FWA:
        np.random.seed(seed_code)
        BS_coords=Prodcting_coords(NumberOfStation,max_Value,min_Value,max_Value,min_Value,decimal)
        FWA_coords=Prodcting_coords(NumberOfFWA,max_Value,min_Value,max_Value,min_Value,decimal)
        
        while np.isin(BS_coords,FWA_coords).any(): ## makesure FWA and BS not repeat
            BS_coords=Prodcting_coords(NumberOfStation,max_Value,min_Value,max_Value,min_Value,decimal)
        # coordinate_random=np.random.uniform(min_Value, max_Value,size=(NumberOfStation,NumberOfFWA))
        # coordinate_random_2 = np.around(coordinate_random,2) # Change to the second decimal place 
        # print(coordinate_random_2)

        ## calculate distance for broadcasting
        FWA_coords = FWA_coords[np.newaxis,:,:]
        BS_coords = BS_coords[:,np.newaxis,:]
        FWA_BS_distance = np.linalg.norm(BS_coords - FWA_coords, axis = -1)
        

        with open(f'{NumberOfStation}BS_{NumberOfFWA}FWA_coords.pkl' , 'wb') as file:
            data = {'FWA_coords': FWA_coords , 'BS_coords': BS_coords , 'FWA_BS_distance': FWA_BS_distance}
            pickle.dump(data, file)
        
        with open(f'{NumberOfStation}BS_{NumberOfFWA}FWA_coords.csv','w',newline='') as csv_file:
            csv_writer = csv.writer(csv_file)

            csv_writer.writerow(['FWA_coords'])
            csv_writer.writerows(FWA_coords.reshape(-1,2))

            csv_writer.writerow(['BS_coords'])
            csv_writer.writerows(BS_coords.reshape(-1,2))

            csv_writer.writerow(['FWA_BS_distance'])
            csv_writer.writerows(FWA_BS_distance)

