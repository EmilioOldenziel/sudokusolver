import recognize_sudoku
import sudoku_solver
import json
import sys, os


def get_sudoku_numbers(sudoku):
    numbers = []

    for row in sudoku:
        for number in row:
            if number:
                numbers.append(number)
    
    return numbers

def main():
    with open('tests.json') as tests_file:  
        tests_list = json.load(tests_file)
    for test in tests_list:
        image_name = test['image_name']
        sudoku = test['sudoku']
        enabled = test['enabled']
        expected_numbers = get_sudoku_numbers(sudoku) 
        sudoku = sum(sudoku, [])
        if enabled:
            result = recognize_sudoku.recognize(image_name, expected_numbers)
            if sudoku == result:
                print(image_name + ': pass')
            else:
                print(image_name + ': fail')
                
                print('test input:')
                print(sudoku_solver.print_sudoku(sudoku))
                print('result:')
                print(sudoku_solver.print_sudoku(result))

main()
