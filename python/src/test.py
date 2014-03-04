'''
Created on Feb 28, 2014
test oop

@author: Hao
'''
class test:
    
    name = 'hao'
    
    def __int__(self):
        pass
    
    def set_name(self,new_name):
        self.name = new_name
            
def main():
    test_obj = test()
    
    print test_obj.name        
    test_obj.set_name('helen')
    print test_obj.name

if __name__ == '__main__': main()