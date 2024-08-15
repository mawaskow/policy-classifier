from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from zipfile import ZipFile
from nltk.tokenize import word_tokenize
import glob

def main(input_path):
    filenames = []
    pdf_dict = {}
    if input_path[-4:]== ".zip":
        with ZipFile(input_path) as myzip:
            filenames = list(map(lambda x: x.filename, filter(lambda x: not x.is_dir(), myzip.infolist())))
    else:
        input_path = input_path+"\\**\\*.*"
        for file in glob.glob(input_path, recursive=True):
            filenames.append(file)


    
    # preproces the documents, and create TaggedDocuments
    tagged_data = [TaggedDocument(words=word_tokenize(doc.lower()),
                                tags=[str(i)]) for i,
                doc in enumerate(data)]
    
    # train the Doc2vec model
    model = Doc2Vec(vector_size=20,
                    min_count=2, epochs=50)
    model.build_vocab(tagged_data)
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.epochs)
    
    # get the document vectors
    document_vectors = [model.infer_vector(
        word_tokenize(doc.lower())) for doc in data]
    
    #  print the document vectors
    for i, doc in enumerate(data):
        print("Document", i+1, ":", doc)
        print("Vector:", document_vectors[i])
        print()

if __name__ == '__main__':
    INPUT_DIR= "C:/Users/Allie/Documents/PhD/IrishPoliciesMar24"
    OUTPUT_PTH = "C:/Users/Allie/Documents/GitHub/policy-classifier/populate_corpora/outputs"
    #
    main()
