import recognize_sudoku
import sudoku_solver
import json
import sys, os

def main():
    with open('tests.json') as tests_file:  
        tests_list = json.load(tests_file)
    for test in tests_list:
        image_name = test['image_name']
        sudoku = test['sudoku']
        enabled = test['enabled']
        sudoku = sum(sudoku, [])
        if enabled:
            result = recognize_sudoku.recognize(image_name)
            if sudoku == result:
                print(image_name + ': pass')
            else:
                print(image_name + ': fail')
                
                print('test input:')
                print(sudoku_solver.print_sudoku(sudoku))
                print('result:')
                print(sudoku_solver.print_sudoku(result))

main()