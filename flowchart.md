``` 
mermaid
flowchat
st=>start: Start
e=>end: End
op=>operation: Select PPM picture
cond=>condition: Whether it is p3 type？
enco=>operation: The user enters data want to hide
cond2=>condition: Data entered is legal?
get=>operation: Get a new encoded PPM picture
st->op->cond->enco->cond2->get->e
cond(yes)->enco
cond(no)->e
cond2(yes)->get
cond2(no)->e
```
