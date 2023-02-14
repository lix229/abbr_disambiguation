README:: Anonymized Data Set, Clinical Abbreviations and Acronyms 
Version 1 (October 30, 2012)
Contact information: 
	Natural Language Processing/Information Extraction (NLP/IE) Program 
	Email: nlp-ie@umn.edu

---------------------------------------
Among 440 most frequently occurring abbreviations and acronyms in clinical text, 75 abbreviations and acronyms had a dominant sense with less than 95% prevalence in 500 random samples. For each, the 500 instances were anonymized and the senses (meanings) were manually annotated and are available for researchers and the larger community. 

Samples were anonymized using the safe harbor method. The basic format of identification codes is as _%#IDENTIFIER#%_ in data. Identifiers were replaced with the following identification codes:

Identifiers 1: Name
Identification codes: _%#NAME#%_

Identifiers 2: Street address (Geographic subdivisions)
Identification codes: _%#STREET#%_

Identifiers 3: City (Geographic subdivisions)
Identification codes: _%#CITY#%_

Identifiers 4: County (Geographic subdivisions)
Identification codes: _%#COUNTY#%_

Identifiers 5: Precinct (Geographic subdivisions)
Identification codes: _%#PRECINCT#%_

Identifiers 6: All geographic subdivisions smaller than a State (Geographic subdivisions)
Identification codes: _%#ADDRESS#%_

Identifiers 7: Zip code (Geographic subdivisions), The geographic unit formed by combining all zip codes with the same three initial digits contains more than 20,000 people
Identification codes: 55455 => _%#55400#%_

Identifiers 8: Zip code (Geographic subdivisions), All such geographic units containing 20,000 or fewer people
Identification codes: _%#00000#%_

Identifiers 9: Dates for under 89 (keep real year), All elements of dates (except year) for dates directly related to an individual, including birth date, admission date, discharge date, date of death
Identification codes: 02/07/2000 => _%#DDMM2000#%_ 
Identification codes: Feb, 07, 2000 => _%#MM#%_, _%#DD#%_, _%#2000#%_
Identification codes: 10/17 => _%#MMDD#%_

Identifiers 10: Dates for over 89
Identification codes: _%#DDMM1914#%_
Identification codes: Feb, 07, 1997 => _%#MM#%_, _%#DD#%_, _%#1914#%_

Identifiers 11: Telephone numbers
Identification codes: _%#TEL#%_

Identifiers 12: Fax numbers
Identification codes: _%#FAX#%_

Identifiers 13: Electronic mail addresses
Identification codes: _%#EMAIL#%_

Identifiers 14: Social security numbers
Identification codes: _%#SSN#%_

Identifiers 15: Medical record numbers
Identification codes: _%#MRN#%_

Identifiers 16: Health plan beneficiary numbers
Identification codes: _%#HPBN#%_

Identifiers 17: Account numbers
Identification codes: _%#ACCOUNTN#%_

Identifiers 18: Certificate/license numbers
Identification codes: _%#LN#%_

Identifiers 19: Vehicle identifiers and serial numbers, including license plate numbers
Identification codes: _%#VN#%_

Identifiers 20: Device
Identification codes: _%#DEVICE#%_

The anonymized acronym and abbreviation sentence data set is pipe delimitated '|'. All whitespaces in sentences were replaced with spaces.

Column 1: The targeted abbreviation or acronym (Short form)
Column 2: Sense, the meaning of the targeted abbreviation or acronym (Long form)
	* If the targeted abbreviation or acronym means the name of a person, it is represented as "NAME".
	* If the targeted abbreviation or acronym means a general English term rather than a clinical term, it is represented as "GENERAL ENGLISH".
	* If the targeted abbreviation or acronym is an error, typo or misused sense, it is represented as "MISTAKE:correct expression". If the annotator can get correct meaning of the misused sense based on the given sentence fragments, it is represented as "Long form:correct acronym or abbreviation". 
	* If the annotator is not sure the sense based on the given sentence fragments, it is represented as "UNSURED SENSE".
Column 3: Representation of the targeted abbreviation or acronym in the given sentence
Column 4: The start position of the targeted abbreviation or acronym in the given sentence. The sentence starts from position 0.
Column 5: The end position of the targeted abbreviation or acronym in the given sentence. The sentence starts from position 0.
Column 6: The section information of the targeted abbreviation or acronym. Heuristic rules were used to detect section information such as:
	* The previous line is an empty line or white space (space or return symbol) only.
	* The position of section phrase starts at the beginning of a line (position 0).
	* The section phrase contains the section indicator symbol ':'.
	* The words of the section phrase from the beginning to the section indicator symbol in the line are written in upper case characters, numbers, and symbols only.
Column 7: Anonymized sample including the targeted abbreviation or acronym 

---------------------------------------------------
Addendum (January 31, 2013)
a. We replaced from 'de-identified' to 'anonymized'