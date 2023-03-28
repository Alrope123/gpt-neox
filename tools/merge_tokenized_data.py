import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
from megatron.data import indexed_dataset

def main():
    file_path = "data"
    base_name = "tokenized_data_text_document"

    # Iterate through all the files and get the name of them
    file_names = os.listdir(file_path)
    file_names = [file[:-4] for file in file_names if file.startswith("tokenized_data_text_document") and '-' in file and file.endswith('.bin')]
    
    # Get the doc_num and bin num for each of these files
    num_to_file = {}
    for fn in file_names:
        doc_num = fn.split('_')[-2].split('-')[0]
        bin_num = fn.split('_')[-1]
        assert doc_num and bin_num
        assert doc_num.isnumeric() and bin_num.isnumeric()
        if doc_num not in num_to_file:
            num_to_file[doc_num] = {bin_num:fn}
        else:
            num_to_file[doc_num][bin_num] = fn

    # Starting from an empty build and merge all of the files
    builder = indexed_dataset.make_builder(
        os.path.join(file_path, base_name + ".bin"),
        impl="mmap",
        vocab_size=50304,
    )

    # Starting merging by the order of doc_num and bin_num
    try:
        sorted_doc_nums = sorted(num_to_file.keys(), key=lambda x: int(x))
        for doc_num in sorted_doc_nums:
            sorted_bin_nums = sorted(num_to_file[doc_num].keys(), key=lambda x: int(x))
            print("Processing documents from doc: {}".format(doc_num))
            for bin_num in sorted_bin_nums:
                file = os.path.join(file_path, num_to_file[doc_num][bin_num])
                print(file)
                # Actual merging
                builder.merge_file_(file)
                # # Remove the old file
                # os.remove(file)
                builder.finalize(os.path.join(file_path, base_name + ".idx"))
        builder.close()
    except Exception as e:
        print("Error:")
        print(e.__doc__)
        builder.finalize(os.path.join(file_path, base_name + ".idx"))
        builder.close()


if __name__ == "__main__":
    main()