# Databricks notebook source

# COMMAND ----------

from bs4 import BeautifulSoup
import re
import regex as reg
import pandas as pd
from unidecode import unidecode
import numpy as np
from sentence_transformers import SentenceTransformer, util
import spacy
from spacy.language import Language
from spacy.lang.en import English

import pyspark.sql.functions as f
import pyspark.sql.types as t

import fuzzysearch

# COMMAND ----------

model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')

# COMMAND ----------

class CreditParser:
    """
    This pipeline serves to structure CRediT author statements (https://www.elsevier.com/researcher/author/policies-and-guidelines/credit-author-statement)
    into author auid:list(contributor roles) for a given PII. 

    For example,
    
    00000000: Writing – original draft, Investigation, Methodology, Data curation, Writing – review & editing. 11111111: Supervision, Conceptualization, Writing – original draft, Methodology, Writing – review & editing, Funding acquisition, Project administration. 222222222: Investigation, Methodology, Formal analysis. 333333333: Investigation, Methodology. 444444444: Investigation, Methodology.
    
     will be structured into:
        [
            (00000000, ['Writing – original draft', 'Investigation', 'Methodology', 'Data curation', 'Writing – review & editing']),
            (11111111, ['Supervision', 'Conceptualization', 'Writing – original draft', 'Methodology', 'Writing – review & editing', 'Funding acquisition', 'Project administration']),
            (222222222, ['Investigation', 'Methodology', 'Formal analysis']),
            (333333333, ['Investigation', 'Methodology']),
            (444444444, ['Investigation', 'Methodology'])
        ]

    This pipeline uses fulltext XML to identify, extract, and parse author contributor statements.
    """
    def __init__(self, au, cs, ag):
        """
        Read in xml using bs4.
        Note that the XML must be structured according to the Elsevier Journal Article DTD,
        available here: https://www.elsevier.com/researcher/author/policies-and-guidelines/elsevier-xml-dtds-and-transport-schemas#Schema-5.15

        """
        # todo - what are the expected inputs here?

        self.author_info = [author for author in au if author if author.values() is not None] #collect author information from dataframe, remove empty dicts

        self.author_group = BeautifulSoup(ag, 'xml') #find author section in xml

        self.authors = self.get_author_names_auids() #collect names, initials, and auids
        self.credit_statement = cs #collect credit statement

        self.auids = [str(auid['auid']) for auid in self.authors] #collect list of auids
        # self.found_duplicate_initials = self.check_duplicate_initials() #check if any author initials are duplicates          

    def check_contributor_role(self):
        """
        Check if contributor roles already structured. If it is, capture names and contributor roles

        If contributor-role xml tag exists:
            -Collect all author sections
            -Loop through authors
                -extract full name
                -match name in self.author dictionary
                -collect all defined contributor roles
                -append auid, contributor role pairs to list
            -return author_roles list(dict)
                        [
                            (57469262800, ['Writing – original draft', 'Investigation', 'Methodology', 'Data curation', 'Writing – review & editing']),
                            (57469262900, ['Investigation', 'Methodology'])
                        ]
        Else return None
        """

        if self.author_group.find('contributor-role'):
            authors = self.author_group.find_all('author')
            author_roles = []
            for author in authors:
                fname = author.find('given-name').text
                lname = author.find('surname').text
                full_name = fname + ' ' + lname
                auid = next(item['auid'] for item in self.authors if item["Name"][0] == full_name)
                roles = [role.text for role in author.find_all('contributor-role')]
                author_roles.append((auid, roles))
            return author_roles
        else:
            return None

    def check_duplicate_initials(self):
        """
        Unresolved issue - how to treat cases where authors have same initials
        This method is a temporary solution to tag PIIs where duplicate initials exist

        Find all initials list in self.authors and flatten
        Compare if list and set(list) have same length
        If != then duplicates exist, return True
        """

        initials = [author['Initials'] for author in self.authors]
        flat_list = [item for initial in initials for item in initial] 
        no_dups = set(flat_list)
        if len(flat_list) != len(no_dups):
            return True
        else:
            return False

    def get_full_name_from_xml(self):
        """
        For some reason, "Au" data from Scopus dataset does not always contain a first and last name.
        This method is a fallback method to extract full name from xml

        Collect all author sections from xml
        Loop through author sections and extract fullnames from given-name + surname tags
        """
        authors = self.author_group.find_all('author')
        author_names = []
        for author in authors:
            fname = author.find('given-name').text
            lname = author.find('surname').text
            author_names.append((fname, lname))
        return author_names
    
    def get_initials(self, au):
        """
        Extracts author initials depending on source.

        Sometimes author middle names are part of 'given_name'
        Ex: given_name = 'John A.' surname = 'Smith'
        First join entire name and then split to get appropriate parts: ' '.join(['John A.', 'Smith']) --> 'John A. Smith' --> ['John', 'A.', 'Smith']
        Then extract initials
        """

        try:
            name = ' '.join([au['given_name'], au['surname']])
        except:
            name = au[0] + ' ' + au[1]

        name_parts = name.split()
        extracted_initials = [''.join([initial[0] for initial in name_parts])]
        extracted_initials.append(''.join([name_parts[0][0], name_parts[-1][0]]))
        extracted_initials = list(set(extracted_initials))

        return extracted_initials

    def get_name(self, name):
        """
        Similar as self.get_initials but retrieving full name with and without middle initial
        """
        name_as_is = ' '.join([name['given_name'], name['surname']])
        split_name = name_as_is.split()
        no_middle_initial = split_name[0] + split_name[-1]
        return [name_as_is, no_middle_initial]

    def get_author_names_auids(self):
        """
        Extracts name, initials, and auid
        Try
            -To extract from scopus data directly
        Except sometime given-name does not exist
            -So use self.get_full_name_from_xml to get author full name instead
        Returns
            [
                {'Name': ['John A. Smith', 'John Smith'],'Initials': ['J.A.S.', 'JAS'],'auid': 00000000},
                {'Name': ['Jane B. Johnson', 'Jane Johnson'], 'Initials': ['J.B.J.', 'JBJ'], 'auid': 111111111},
                {'Name': ['Jack Jackson'], 'Initials': ['J.J.', 'JJ'], 'auid': 333333333}
            ]
        """
        try:
            author_dict = [{'Name':self.get_name(d), 'Initials': self.get_initials(d), 'auid':d['auid']} for d in self.author_info]
        except:
            author_names = self.get_full_name_from_xml()
            name_matches = []
            for name in author_names:
                for i in self.author_info:
                    if i['surname'] == name[1]:
                        name_matches.append(i)
            author_dict = []
            for author in author_names:
                scopus_inits = [i['initials'] for i in self.author_info if i['surname'] == author[1]]            
                temp_dict = [{'Name': [author[0] + ' ' + author[1]], 'Initials': self.get_initials(author).extend(scopus_inits), 'auid':d['auid']} for d in name_matches if author[1] == d['surname']][0]
                author_dict.append(temp_dict)
        return author_dict
    
    def calclulate_nearest_match(self, target, credit_statement):
        """
        Authors identify themselves differently (A.B. or AB or A.B or AB. or Alpha Bravo or A.Bravo etc)
        To account for variation, use Levenshtein distance via fuzzysearch library to find matches, allowing only a max of 5 insertions.
        No deletions or substitutions allowed. Do not want to match ABC with AB or ABX


        -Remove any spaces that may exist between initials
        -Remove titles like Mr/Mrs, Dr, etc
        -For each author in self.authors, perform fuzzy search between author info (either initials or name depending on target) and the credit statement 
        
        Returns credit statement with names or initials replaced with auids

        """

        initials_with_spaces = re.findall(r"\b[A-Z](?:[A-Z\.][\.\s]*)[A-Z]\b\.?", credit_statement)
        if initials_with_spaces:
            for match in initials_with_spaces:
                strip_match = match.replace(" ", "")
                credit_statement = credit_statement.replace(match, strip_match)

        titles = ("MR","DR","MRS","PROF","MS", "DRS")
        ptrn = re.compile(r'\b(?:' + '|'.join(titles) + r')(?:\.|\b)\s*', flags=re.I)
        credit_statement = ptrn.sub("", credit_statement)
        credit_statement = credit_statement.replace('&', '').strip()

        matches = []
        for author in self.authors:
            author_match = []
            if author[target]:
                author_form = author[target]
                for a in author_form:
                    fuzzy_match = fuzzysearch.find_near_matches(a, credit_statement, max_deletions=0, max_insertions=5, max_substitutions=0)
                    if fuzzy_match:
                        matches.append(fuzzy_match[0].matched)
                        author_match.append((a, fuzzy_match[0].dist, fuzzy_match[0].matched))
                if author_match:
                    author_match = sorted(author_match, key=lambda i: i[1])
                    credit_statement = credit_statement.replace(author_match[0][2], str(author['auid']))
        pattern = r'\S*?(\d{10,11}\.0)[^\s,]+'
        credit_statement = re.sub(pattern, r'\1', credit_statement)           

        return credit_statement, matches

    def replace_just_last_names(self, text):
        """
        Sometimes author only report the last name in the credit statement. 
        """
        formatted_results = []
        for sentence in text:
            for author in self.authors:
                sentence = sentence.replace(str(author['auid'])+'.', str(author['auid'])) #odd case where name has already been replaced from prior method but leaves period at end
                last_name = author['Name'][0].split()[-1]
                sentence = sentence.replace(unidecode(last_name), str(author['auid']))
            formatted_results.append(sentence)
        return formatted_results

    def replace_reverse_name(self, text):
        """
        In some cases, authors write last name then first name. So (John Smith will not match with Smith John)

        Replace the reverse of the author name with their auid
        """
        formatted_results = []
        for sentence in text:
            for author in self.authors:
                reverse_name = " ".join(author['Name'][0].split(" ")[::-1])
                sentence = sentence.replace(reverse_name, str(author['auid']))
            formatted_results.append(sentence)
        return formatted_results

    def replace_with_auid(self, text, target):

        """
        This method simply uses above calculate nearest match to replace names/initials with auid for all
        credit statements if multiple exist

        For each sentence in credit statement replace with auid depending on target (name or initials)
        Return list of edited credit statements
        """
        text = self.normalize_figs(text) #fix issues with how figures are stated to prevent issues with sentence splitting
        formatted_result = []
        matches = []
        for sentence in text: #for each sentence in the credit statement paragraph, match author info 
            calculate = self.calclulate_nearest_match(target, sentence)
            sentence = calculate[0]
            matches.append(calculate[1])
            formatted_result.append(sentence)
        matches = list(set([item for sublist in matches for item in sublist]))
        return formatted_result, matches

    def check_format_and_replace(self, text):
        """
        Check which format author report their identities - full name or initials
        Depending on format, replace their names with author ID

        This is method should be improved as it requires first checking for matches and if condition met then transform
        """
        #Some errors with extraction of credit statement section includes irrelevant sections. Rough filter to remove these
        text = [i for i in text if not i.startswith(("Financial/nonfinancial disclosures","Other contributions", 
                                                     "Role of sponsors", "Additional information", "Competing Financial Interests", "Disclosure", "Conflicts of Interest"))]

        potential_name_formatted_results = self.replace_with_auid(text, 'Name') #check for instances where full name is reported
        potential_inits_formatted_results = self.replace_with_auid(text, 'Initials') #check for instances where initials are reported

        if any(name in cs for cs in self.credit_statement for name in potential_name_formatted_results[1]): #if names match, take output
            formatted_results = potential_name_formatted_results[0]
        else: #else take initials matches
            formatted_results = potential_inits_formatted_results[0]

        #also need to check for edge cases like if only last names are reported, names are reported in reverse (Last Name, First Name)
        last_names = [a['Name'][0].split()[-1] for a in self.authors]
        first_names = [a['Name'][0].split()[0] for a in self.authors]
        #if any last name matches but no first names
        if any(name in cs for cs in formatted_results for name in last_names) and not any(name in cs for cs in formatted_results for name in first_names):
            formatted_results = self.replace_just_last_names(formatted_results)
        #reverse the name and look for matches
        if any(" ".join(name.split(" ")[::-1]) in cs for cs in formatted_results for name in [i['Name'][0] for i in self.authors]):
            formatted_results = self.replace_reverse_name(formatted_results)
        # I think I had this because of the 'not any(name ...)' part above. Sometimes there were cases where some authors reported first and last name, for
        # example when authors have duplicate initials. But the rest of the authors had just last name. So not any == False
        if any(name in cs for cs in formatted_results for name in last_names):
            formatted_results = self.replace_just_last_names(formatted_results)
            
        
        return formatted_results

    def split_comma_separated_statements(self, text):
        """
        Some author have grammatical errors in which each sentence is separated by a comma
        rather than a period or semi-colon

        Ex: John Smith worked on the draft of the paper, Jane Smith ran the analysis
        This would not split correctly into two sentences

        Identify if credit statement matches this pattern and split sentences correctly
        """

        result = []
        pattern = r'(?<=\d{10}\s[A-Za-z\s–,-]+(?<!and))(?=\s\d{10}|\.$)'
        for i in text:
            sentences = reg.split(pattern, i)
            sentences = [sentence.strip().rstrip('.') for sentence in sentences]
            result.extend(sentences)
        return list(filter(None, result))

    def normalize_figs(self, text):
        """
        Some author report which figures they contributed to, creating issues for sentence splitting
        For example:
            'Author A worked on Fig. 1.' would split into ['Author A work on Fig.', '1.']

        As a work around, identify these cases and remove space between Fig. and 1. --> Fig.1. which resolves
        incorrect sentence tokenization
        """

        pattern = r'(?:Fig\.?|Figs\.?|Figure\.?||Figures\.?)\s*(?:\d+(?:[-–]\d+)?(?:,\s*)?)+'
        result = []
        for sentence in text:
            matches = re.findall(pattern, sentence)
            no_space = []
            for match in matches:
                no_space.append((match, match.replace(' ', '')))
            for k,v in no_space:
                sentence = sentence.replace(k, v)
            result.append(sentence)
        return result

    def validate_text(self, text):
        """
        Checks whether all sentences contain an author identifier. Returns true if
        any sentence does not contain an auid.
        """

        for sentence in text:
            if any(author in sentence for author in self.auids):
                continue
            else:
                return True

    def map_author_to_phrase(self, text): #this doesn't cover if first sentence doesn't have authors listed
        """
        Some edge cases have the following format:
            'John Smith: Conceived and designed the experiments;
             Performed the experiments; Analyzed and interpreted the data; 
             Contributed reagents, materials, analysis tools or data; Wrote the paper.'

        where the author is only listed at the beginning and all contributions are written as separate phrases.
        This pipeline splits on periods and semicolons, so these phrases would not have auids associated.

        This method identifies phrases without author identifiers, saves auids from previous phrase and prepends to candidate sentences
        """
        if self.validate_text(text):
            mapped_author = []
            para_authors = []
            for sentence in text:
                
                if any(author in sentence for author in self.auids):
                    para_authors = [] #collect authors that are found in the sentence
                    for author in self.auids:
                        if author in sentence:
                            para_authors.append(author)
                    mapped_author.append(sentence)
                else:
                    combine = f"{', '.join(para_authors)} {sentence}" #if a sentence doesn't contain an author identifier, prepend authors from para_authors to sentence
                    mapped_author.append(combine)
            return mapped_author
        else:
            return text

    def check_for_all_author_statement(self, text):
        """
        Some authors include a statement such as 'All authors contributed equally to manuscript writing'.
        In this instance, 'All authors' should be mapped to all auids. This method uses sentence transformers
        to find statements that are semantically similar to the phrase 'All authors contributed'. If cosine similarity
        score is above 0.6, auids are pre-prended to sentence.
        """
        # model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2') #model defined at top of notebook to ensure model is broadcasted to spark workers
        input_text = [t for t in text if not any(author in t for author in self.auids)] #  filter for sentences that don't contain author identifiers
        if not input_text: #if there aren't any, just return input text
            return text
        other_text = [t for t in text if any(author in t for author in self.auids)]
        all_authors_statement = 'All authors contributed'
        all_authors_query = model.encode([all_authors_statement], convert_to_tensor=True) #embed query using all-MiniLM-L12-v2 
        if len(input_text) == 1:
            input_text.append('') # I think I did this because sentencetransformers requires a batch > 1 or raises error
            # candidate_sentence = model.encode(input_text, convert_to_tensor=True)
            # hits = util.cos_sim(all_authors_query, candidate_sentence)
        candidate_sentences = model.encode(input_text, convert_to_tensor=True) #embed sentences that should be checked for All authors contributed
        hits = util.semantic_search(all_authors_query, candidate_sentences) #calculate cosine similarity
        hits = hits[0]
        for hit in hits:
            if hit['score'] > 0.60: #loop through results, filtering for matches cos sim > 0.6
                all_match = input_text[hit['corpus_id']]
                input_text[hit['corpus_id']] = ', '.join(self.auids) + ' ' + all_match #retrieve text and prepend all author ids to it
        input_text = list(filter(None, input_text)) #filter out none - the input_text.append('') from above
        output = other_text + input_text
        return output

    
    def split_sentences(self):
        """
        Tokenize credit statement into sentences

        -Collect credit statement
        -Normalize how contribution to figures are displayed
        -Use Spacy to split sentences, adding custom split including semi-colons
        -Replace names/initials with auids
        -Check for incorrect comma separated credit statement and fix
        -Cehck for statements that claim all authors contributed to a role
        -Fix instances where trailing sentences refer back to author in prior sentence
        """

        text = self.credit_statement
        text = self.normalize_figs(text)
        
        nlp = spacy.blank('en')
        config = {"punct_chars":[".", "?", "!", ";"]}
        nlp.add_pipe("sentencizer", config=config)
        doc = nlp.pipe(text)
        paras = list(doc)
        text = [[sent.text for sent in para.sents] for para in paras]
        text = [item.strip() for sentence in text for item in sentence]

        text = self.check_format_and_replace(text)

        text = self.split_comma_separated_statements(text)

        text = self.check_for_all_author_statement(text)

        text = self.map_author_to_phrase(text)

        return text        
        
    def parse_statement(self):
        """
        Parse the credit statement into auid, list(contributor roles)

        First, if structured contribution role exists, return that
        Else
            -Split the credit statement sentences
            -For each sentence in credit statement, extract auids and everything else using regex
            -Create auid, list(contributor roles) pairs
            -Append to list
        """

        if self.check_contributor_role() is not None:
            # return self.check_contributor_role(), 'Structured'
            return self.check_contributor_role()
        
        doc = self.split_sentences()
        results = []
        for sent in doc:
            auids = re.findall(r'\d{9,11}\.0', sent)
            contributions = re.findall(r',?[a-zA-Z][a-zA-Z]*,?', sent)
            contributions = ' '.join(contributions)
            if contributions.startswith('and'): #resolve cases where auids are separated by 'and' and remove
                contributions = contributions[4:]
            if contributions:
                author_contribution = [(auid, contributions) for auid in auids]
                results.append(author_contribution)
        results = list(set([item for sublist in results for item in sublist]))
        # return results, 'Unstructured'
        return results




# COMMAND ----------

#load data with pre-extracted credit statements
cs_unstructured_corpus_interim = table("icsr.sd_xml_enriched_unstructured_interim")

def run_pipeline(au, cs, ag):
    try:        
        credit_parser_instance = CreditParser(au, cs, ag)
        results = credit_parser_instance.parse_statement()
        return results
    except Exception as e:
        output = [['Could not process']]
    return output

#custom pyspark pandas UDF
@f.pandas_udf(t.ArrayType(t.ArrayType(t.StringType())))
def pipeline_udf(au_col, cs_col, ag_col):
    return pd.Series([run_pipeline(au, cs, ag) for au, cs, ag in zip(au_col, cs_col, ag_col)])

cs_unstructured_corpus_output = cs_unstructured_corpus_interim.withColumn("output", pipeline_udf(cs_unstructured_corpus_interim.Au, cs_unstructured_corpus_interim.credit_statement, cs_unstructured_corpus_interim.author_group))
cs_unstructured_corpus_output.write.mode("overwrite").format("delta").saveAsTable("database.table_name")

# COMMAND ----------

errors = table("icsr.sd_xml_enriched_unstructured_processed_final")
errors.where(f.array_contains(f.col('output')[0], 'Could not process')).count()