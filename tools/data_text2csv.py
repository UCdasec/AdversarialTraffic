import pandas as pd
import os



def read_txt(path):
    "extract the packet direction from txt"

    trace = []
    f = open(path,mode='r')
    line = f.readline()
    while line:
        data = line.split()
        trace.append(int(data[1:2][0]))
        line = f.readline()
    f.close()

    return trace


def write2csv(data,labels,outputpath):
    "write packet direction of each trace into csv"
    data = pd.DataFrame(data)
    # data = data.fillna(0)
    data['label'] = labels
    print(data.head(-5))
    data.to_csv(outputpath,index=0)
    print('writing ...{} done! '.format(outputpath))


def get_monitored_data(input_path,outpath):
    "extract monitored info from txt and write to csv"
    "from file 0-0 to 99-89"

    data = []
    labels = []
    for i in range(100):
        for j in range(90):
            file = str(i) + '-' + str(j)
            data_path = input_path + file
            print('processing file {}'.format(data_path))
            trace = read_txt(data_path)
            labels.append(i)
            data.append(trace)
    outpath = outpath + 'wang_Mon.csv'
    write2csv(data,labels,outpath)



def get_unmonitored_data(inpath,outpath):
    "extract unmonitored file from txt and write it csv"
    "from file 0 to 8999"

    data = []
    labels = []
    for i in range(9000):
        file = str(i)
        data_path = inpath + file
        print('processing file {}'.format(data_path))
        trace = read_txt(data_path)
        labels.append(100)
        data.append(trace)
    outpath = outpath + 'wang_UnMon.csv'
    write2csv(data,labels,outpath)






if __name__ == '__main__':
    input_path = '../data/wf_wang/batch/'
    out_path = '../data/wf_wang/'

    # get_monitored_data(input_path,out_path)
    get_unmonitored_data(input_path,out_path)
