from tkinter import *
from ttkwidgets.autocomplete import AutocompleteEntry

# Sample list of symptoms (replace this with actual symptoms)
symptoms = [
    'Fever', 'Cough', 'Fatigue', 'Shortness of breath', 'Headache', 'Sore throat',
    'Loss of taste or smell', 'Muscle or body aches', 'Nausea or vomiting', 'Diarrhea'
]

root = Tk()
root.title('Disease Diagnostic')
root.geometry('700x700')

# Frame setup
frame = Frame(root, bg='#ffffff')
frame.pack(fill=BOTH, expand=True)

left_frame = Frame(frame, width=200, height=400)
left_frame.grid(row=0, column=0, padx=10, pady=5, sticky='nsew')

Label(
    left_frame,
    bg='#f25252',
    font=('Times', 21),
    text='Symptoms List'
).grid(row=0, column=0, pady=10)

tool_bar = Frame(left_frame, width=180, height=185, bg="purple")
tool_bar.grid(row=1, column=0, padx=5, pady=5)

# Entry widget with autocomplete
entry = AutocompleteEntry(
    left_frame,
    width=30,
    font=('Times', 18),
    completevalues=symptoms
)
entry.grid(row=2, column=0, padx=5, pady=5)

def add_symptom():
    symptom = entry.get()
    if symptom and symptom not in listbox.get(0, END):
        listbox.insert(END, symptom)
        entry.delete(0, END)

def diagnose():
    selected_symptoms = listbox.get(0, END)
    if not selected_symptoms:
        result_label.config(text="No symptoms added")
    else:
        result_label.config(text="Diagnosis based on symptoms: " + ", ".join(selected_symptoms))

# Buttons
add_button = Button(left_frame, text="Add", command=add_symptom)
add_button.grid(row=3, column=0, padx=5, pady=5)

diagnose_button = Button(left_frame, text="Diagnose", command=diagnose)
diagnose_button.grid(row=4, column=0, padx=5, pady=5)

# Listbox to display added symptoms
listbox = Listbox(left_frame, width=30, height=10, font=('Times', 14))
listbox.grid(row=5, column=0, padx=5, pady=5)

# Label to show diagnosis result
result_label = Label(left_frame, text="", font=('Times', 14))
result_label.grid(row=6, column=0, padx=5, pady=10)

root.mainloop()
