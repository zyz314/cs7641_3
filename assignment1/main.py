from datasets.uciml import AdultDataset

def main():
    dataset = AdultDataset()
    for i, (x, y) in enumerate(dataset):
        print(i, x, y)

if __name__ == '__main__':
    main()
