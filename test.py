def reassign(list):
  list = [0, 1]


def append(list):
  list.append(1)




if __name__ == '__main__':
    
    listi = [0]+[0]*5
    print(listi)
    reassign(listi)
    print(listi)
    append(listi)
    print(listi)