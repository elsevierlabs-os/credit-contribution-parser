# Databricks notebook source
import pyspark.sql.functions as f
import pyspark.sql.types as t
from bs4 import BeautifulSoup
import unicodedata
import html

# COMMAND ----------

"""
Table contains the following info
Publication ID
Author Info: Array[{"author ID":"value", "given name", "initials", "last name}]
xml: full text xml
"""
cs_unstructured_corpus = table("database.table")

# COMMAND ----------

def get_credit_statement(xml):
    """
    Find credit statement using regex, collect parent section 2 levels up from body
    Some credit statements are broken up in to multiple paragraphs
    """
    soup = BeautifulSoup(xml, 'xml')

    try:
        credit_statement_sec = soup.body.find(string= re.compile('CRediT|[Aa]uthor(ship)? ([Cc]ontribut|[Ss]tatement)|([Cc]ontribution|[Cc]redit) [Ss]tatement')).parent.parent
        paras = credit_statement_sec.find_all('para', {"view":"all"})
        if not paras:
            raise Exception('need to go up a level')
    except:
        credit_statement_sec = soup.body.find(string= re.compile('CRediT|[Aa]uthor(ship)? ([Cc]ontribut|[Ss]tatement)|([Cc]ontribution|[Cc]redit) [Ss]tatement')).parent.parent.parent
        paras = credit_statement_sec.find_all('para', {"view":"all"})
    credit_statement = [item.text.replace(u'\xa0', u' ') for item in paras] #weird encoding error
    credit_statement = [cs.replace(':', '') for cs in credit_statement]
    return credit_statement

def try_cs(xml):

    try:
        output = get_credit_statement(xml)
    except Exception as e:
        output = ''.join(traceback.format_exception(None, e, e.__traceback__))
    return output


def get_author_group(xml):
    soup = BeautifulSoup(xml, 'xml')
    author_group = soup.find('author-group')
    author_group = html.unescape(unicodedata.normalize("NFKD",str(author_group)))
    return author_group

# COMMAND ----------

#collect credit statement section from xml
cs_udf = f.udf(try_cs, t.ArrayType(t.StringType()))
cs_unstructured_corpus = cs_unstructured_corpus.withColumn('credit_statement', cs_udf(cs_unstructured_corpus.xml))

#collect author information from xml
ag_udf = f.udf(get_author_group, t.StringType())
cs_unstructured_corpus = cs_unstructured_corpus.withColumn('author_group', ag_udf(cs_unstructured_corpus.xml))

cs_unstructured_corpus = cs_unstructured_corpus.select(['PII', 'Au', 'credit_statement', 'author_group'])
cs_unstructured_corpus.write.mode("overwrite").format("delta").saveAsTable("database.table")