from datasets.uciml import AdultDataset

def main():
    dataset = AdultDataset()
    for i, sample in enumerate(dataset):
        print(i, sample)

if __name__ == '__main__':
    main()
