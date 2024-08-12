from tkinter import *
from tkinter.font import Font
from ttkwidgets.autocomplete import AutocompleteEntry

# Sample list of symptoms (replace this with actual symptoms)
symptoms = [
    'Fever', 'Cough', 'Fatigue', 'Shortness of breath', 'Headache', 'Sore throat',
    'Loss of taste or smell', 'Muscle or body aches', 'Nausea or vomiting', 'Diarrhea'
]


def add_symptom():
    symptom = entry.get()
    if symptom and symptom not in listbox.get(0, END):
        listbox.insert(END, symptom)
        entry.delete(0, END)

def populate_text_area(text_widget, items):
    """Populates the text widget with items from the list."""
    text_widget.delete(1.0, END)  # Clear existing content
    for item in items:
        text_widget.insert(END, item + '\n')


root = Tk()
root.title('Disease Diagnostic Tool')
root.geometry('1280x720')
#menu bar 
menu_bar = Menu(root)
root.config(menu=menu_bar)


menuitem1= Menu(menu_bar, tearoff = 0)
menu_bar.add_cascade(label ='Edit', menu = menuitem1) 
menuitem1.add_command(label ='Cut', command = None) 

headingLabel = Label(root, text='Disease Diagnostic Tool', font=('times new roman', 30, 'bold')
                     , bg='gray20', fg='gold', bd=12, relief=GROOVE)
headingLabel.pack(fill=X)



customer_details_frame = LabelFrame(root, text='Patient Details', font=('times new roman', 15, 'bold'),
                                    fg='gold', bd=8, relief=GROOVE, bg='gray20')
customer_details_frame.pack(fill=X)

nameLabel = Label(customer_details_frame, text='Name', font=('times new roman', 15, 'bold'), bg='gray20',
                  fg='white')
nameLabel.grid(row=0, column=0, padx=20)

nameEntry = Entry(customer_details_frame, font=('arial', 15), bd=7, width=18)
nameEntry.grid(row=0, column=1, padx=8)

phoneLabel = Label(customer_details_frame, text='Phone Number', font=('times new roman', 15, 'bold'), bg='gray20',
                   fg='white')
phoneLabel.grid(row=0, column=6, padx=20, pady=2)

phoneEntry = Entry(customer_details_frame, font=('arial', 15), bd=7, width=18)
phoneEntry.grid(row=0, column=7, padx=8)


AgeLabel = Label(customer_details_frame, text='Age', font=('times new roman', 15, 'bold'), bg='gray20',
                   fg='white')
AgeLabel.grid(row=0, column=4, padx=20, pady=2)

AgeEntry = Entry(customer_details_frame, font=('arial', 15), bd=7, width=10)
AgeEntry.grid(row=0, column=5, padx=8)

AgeLabel = Label(customer_details_frame, text='Gender', font=('times new roman', 15, 'bold'), bg='gray20',
                   fg='white')
AgeLabel.grid(row=0, column=2, padx=20, pady=2)

AgeEntry = Entry(customer_details_frame, font=('arial', 15), bd=7, width=10)
AgeEntry.grid(row=0, column=3, padx=8)


frame2= Frame(root)
frame2.pack(fill=X)

#menubar 1
Symptoms_frame = LabelFrame(frame2, text='Symptoms', font=('times new roman', 15, 'bold'),
                                    fg='gold', bd=8, relief=GROOVE, bg='gray20')
Symptoms_frame.grid(row=0, column=1, padx=10)

Symptom_Label = Label(Symptoms_frame, text='ENTER SYMPTOMS', font=('times new roman', 15, 'bold'), bg='gray20',
                  fg='white')
Symptom_Label.grid(row=0, column=0, padx=10)



# Entry widget with autocomplete
entry = AutocompleteEntry(
    Symptoms_frame,
    width=30,
    font=('Times', 18),
    completevalues=symptoms
)
entry.grid(row=2, column=0, padx=5, pady=5)

# Buttons
add_button = Button(Symptoms_frame, text="Add", command=add_symptom)
add_button.grid(row=3, column=0, padx=5, pady=5)

# # Listbox to display added symptoms
listbox = Listbox(Symptoms_frame, width=30, height=10, font=Font(family="times new roman", size=14))
listbox.grid(row=5, column=0, padx=5, pady=5)


# Text area with symptoms

Symptoms_Display_frame = LabelFrame(frame2, text='Symptoms List', font=('times new roman', 15, 'bold'),
                                    fg='gold', bd=8, relief=GROOVE, bg='gray20')
Symptoms_Display_frame.grid(row=0, column=0, padx=5, pady=5)


text_area = Text(Symptoms_Display_frame, wrap=WORD, height=20, width=50,)
text_area.grid(row=0, column=0, padx=10, pady=10)





# Populate the Text widget with the Symptoms list
populate_text_area(text_area, symptoms)

Diagnosis_Display_frame = LabelFrame(frame2, text='Diagnosis', font=('times new roman', 15, 'bold'),
                                    fg='gold', bd=8, relief=GROOVE, bg='gray20')
Diagnosis_Display_frame.grid(row=0, column=3, padx=5, pady=5)


text_area = Text(Diagnosis_Display_frame, wrap=WORD, height=20, width=50,)
text_area.grid(row=1, column=3, padx=10, pady=10)


# Frame setup
# frame = Frame(root, bg='#ffffff')
# frame.pack(fill=BOTH, expand=True)

# left_frame = Frame(frame, width=200, height=400)
# left_frame.grid(row=0, column=0, padx=10, pady=5, sticky='nsew')


# Label(
#     left_frame,
#     bg='#f25252',
#     font=('Times', 21),
#     text='Symptoms List'
# ).grid(row=1, column=0, pady=10)

# tool_bar = Frame(left_frame, width=180, height=185, bg="purple")
# tool_bar.grid(row=1, column=0, padx=5, pady=5)





# def diagnose():
#     selected_symptoms = listbox.get(0, END)
#     if not selected_symptoms:
#         result_label.config(text="No symptoms added")
#     else:
#         result_label.config(text="Diagnosis based on symptoms: " + ", ".join(selected_symptoms))



# diagnose_button = Button(left_frame, text="Diagnose", command=diagnose)
# diagnose_button.grid(row=4, column=0, padx=5, pady=5)



# # Label to show diagnosis result
# result_label = Label(left_frame, text="", font=('Times', 14))
# result_label.grid(row=6, column=0, padx=5, pady=10)

root.mainloop()
