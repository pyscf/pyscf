

def run():

    #x = 1


    def foo():
        nonlocal x
        x = 3
        print(x)

    foo()
    print(x)



run()

