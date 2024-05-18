from datasets.uciml import AdultDataset, DryBeanDataset

def main():
    dataset = DryBeanDataset()
    for i, (x, y) in enumerate(dataset):
        print(i, x, y)

if __name__ == '__main__':
    main()
