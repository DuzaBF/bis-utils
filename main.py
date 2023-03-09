import argparse



def main():
    parser = argparse.ArgumentParser(
                prog = 'BIS utilities')
    parser.add_argument('filename') 
    args = parser.parse_args()

if __name__ == "__main__":
    main()
