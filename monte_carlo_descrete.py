import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import random 


def read_data( filepath):
    df = pd.read_csv(filepath)
    col_data = df['frequency']
    list = col_data.tolist()
    return list

def print_data(list): 
    for x in list:
        print(x)


def random_number_generator(simulation_range):
    rand_list = []
    for x in range(simulation_range):
        rand_list.append(random.randint(0, 999)) 
    return rand_list
    

def calc_total(list):
    sum = 0 
    for x in list:
        sum = sum +x 
    return sum


def frequency_distribution(sum , list):
    distribution_list = []
    for x in list:
        a = x / sum 
        distribution_list.append(a)
    return distribution_list

def assign_random(filepath , newfile , distribution_list ):
    df = pd.read_csv(filepath) 
    main_data = df['x_val']
    col = main_data.tolist()
    start_tag = []
    end_tag = []
    s =0
    for x in distribution_list:
        e = s + int(x*1000) - 1 
        start_tag.append(s)
        end_tag.append(e)
        s = e + 1 
    output_df = pd.DataFrame({
        'x_val': col[:len(start_tag)],  
        'start_tag': start_tag,
        'end_tag': end_tag
    })

    output_df.to_csv(newfile, index = False)

def simulate(random_numb , generate_csv):
    df = pd.read_csv(generate_csv) 
    x_variable = df['x_val'].tolist()
    start_tag = df['start_tag'].tolist() 
    end_tag = df['end_tag'].tolist()
    recorded_data = []
    for y in random_numb:
        for x  in range(len(df)): 
            if start_tag[x] <= y <= end_tag[x]: 
                recorded_data.append(x_variable[x])
    return recorded_data 

def plot_data(random_numb_list , recorded_data):
    for x in range(len(random_numb_list)):
        plt.scatter(random_numb_list[x],recorded_data[x] )
    plt.xlabel("random numbers")
    plt.ylabel("frequency")
    plt.savefig("plotted.png")

def animated_plot():
    pass

def main(): 
    frequency = read_data('./data.csv') 
    random_numbers = random_number_generator(10000)
    sum = calc_total(frequency)
    distribute = frequency_distribution(sum , frequency)
    assign_random('./data.csv' , "./generated.csv", distribute)
    recorded_data = simulate(random_numbers, './generated.csv')
    print(random_numbers)
    print(recorded_data)
    plot_data(random_numbers , recorded_data)


if __name__ == "__main__":
    main()


