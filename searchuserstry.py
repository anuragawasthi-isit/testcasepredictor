from gensim.models import Doc2Vec

from main import acptcrtlst, clean_text



# Function to convert
def listToString(s):
    # initialize an empty string
    str1 = "  "

    # traverse in the string
    for ele in s:
        str1 += " " + ele

    # return string
    return str1




def findsimilartc(acptid):
    model = Doc2Vec.load("basic1.model")
    #print(acptcrtlst[0])
    tokens = [acptid]
    print(listToString(tokens))
    # tokens = tokens.remove("Given")

    new_vector = model.infer_vector(listToString(tokens).split())

    MOST_SIMILAR_TCS = model.docvecs.most_similar([new_vector])
    accuracy = [tc_tuple[1] for tc_tuple in MOST_SIMILAR_TCS]
    similar_tcs = [tc_tuple[0] for tc_tuple in MOST_SIMILAR_TCS]
    return (similar_tcs,accuracy)
def findsimilartcwitacc(acptid):
    model = Doc2Vec.load("basic1.model")
    #print(acptcrtlst[0])
    tokens = [acptid]
    print(listToString(tokens))
    # tokens = tokens.remove("Given")

    new_vector = model.infer_vector(listToString(tokens).split())

    MOST_SIMILAR_TCS = model.docvecs.most_similar([new_vector])
    accuracy = [tc_tuple[1] for tc_tuple in MOST_SIMILAR_TCS]
    similar_tcs = [tc_tuple[0] for tc_tuple in MOST_SIMILAR_TCS]
    return (similar_tcs)