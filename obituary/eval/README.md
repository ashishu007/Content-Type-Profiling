# Obituary Error Annotations

## Directories

* [`annots\`](./annots): accuracy error annotations
* [`csvs\`](./csvs): automated scores in CSV format
* [`jsons\`](./jsons): automated scores in JSON format
* [`txts\`](./txts): system generated texts used for accuracy evaluation

## Error Categories

1. **Name**: Name of person, child, grandchild, place, etc.
2. **Number**: Date, time, or other number.
3. **Word**: 
    * Gender error, such as: wife written to male person; father written to female; etc. 
    * If the information about child/grandchild is not given and the summary mentions father/mother etc, it is word error.
    * Info longer than a word are also word error.
4. **Context**: N/A
5. **Not Checkable**: N/A
6. **Other**: N/A

## Some patterns

* Neural is not making errors but missing out some information. For example, it sometimes misses the name of relatives (grandchild, great-grandchild name) of the deceased person.
* Hallucinations are also possible in neural system: adding completely different/new name of cemetary place, even though its not given or different from the one given in the input.
* Neural also adds some different words such as: all *mourners* are welcome to funeral; *interment* will take place at 1 pm. These words rarely appear in cbr, that too in specific conditions.
* CBR is only making errors when the attribute value is not given. For example, a template selected from the case-base displays grandchild name of the deceased person, but the input doesn't have that information. In that case, the error appears in the template, where the *grandchild_name* attribute value is missing.

