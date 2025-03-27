import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from groq import Groq
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import textwrap

# Initialize Groq API client (replace with your key)
client = Groq(api_key="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")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
    except Exception as e:
        return f"Error reading {pdf_path}: {str(e)}"

# 1. Plagiarism Detection
def detect_plagiarism(reference_path, pdf_paths, gui):
    ref_text = extract_text_from_pdf(reference_path)
    vectorizer = TfidfVectorizer()
    results = []
    total = min(len(pdf_paths), 50)
    
    for i, pdf_path in enumerate(pdf_paths[:50]):
        text = extract_text_from_pdf(pdf_path)
        if "Error" not in text:
            tfidf_matrix = vectorizer.fit_transform([ref_text, text])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            plagiarism_percentage = similarity * 100
            results.append(f"{pdf_path}: {plagiarism_percentage:.2f}%")
        else:
            results.append(text)
        progress = (i + 1) / total * 100
        gui.update_progress(f"Processing Plagiarism: {progress:.1f}%")
        gui.root.update()
    
    return results

# 2. Document Classification
def train_classifier():
    train_texts = [
        "Revenue increased by 10% this quarter.",
        "Patient diagnosed with hypertension.",
        "Contract signed on January 1st."
    ]
    train_labels = ["financial", "healthcare", "legal"]
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_texts)
    clf = LogisticRegression()
    clf.fit(X_train, train_labels)
    return clf, vectorizer

def classify_documents(pdf_paths, clf, vectorizer, gui):
    results = []
    total = min(len(pdf_paths), 50)
    
    for i, pdf_path in enumerate(pdf_paths[:50]):
        text = extract_text_from_pdf(pdf_path)
        if "Error" not in text:
            X_test = vectorizer.transform([text])
            category = clf.predict(X_test)[0]
            results.append(f"{pdf_path}: {category}")
        else:
            results.append(text)
        progress = (i + 1) / total * 100
        gui.update_progress(f"Processing Classification: {progress:.1f}%")
        gui.root.update()
    
    return results

# 3. Database Question-Answering (DBQA)
def query_documents(pdf_paths, question, gui):
    results = []
    errors = []
    total = min(len(pdf_paths), 50)
    
    for i, pdf_path in enumerate(pdf_paths[:50]):
        text = extract_text_from_pdf(pdf_path)
        if "Error" not in text:
            prompt = f"Document: {text}\nQuestion: {question}\nAnswer:"
            try:
                response = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama3-8b-8192"
                )
                answer = response.choices[0].message.content
                wrapped_answer = "\n".join(textwrap.wrap(answer, width=80))
                results.append((pdf_path, wrapped_answer))
            except Exception as e:
                errors.append(f"{pdf_path}: Error - {str(e)}")
        else:
            errors.append(text)
        progress = (i + 1) / total * 100
        gui.update_progress(f"Processing DBQA: {progress:.1f}%")
        gui.root.update()
    
    return results, errors

# GUI Class
class DocumentAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Document Analyzer Pro")
        self.root.geometry("1200x900")
        self.root.configure(bg="#ffffff")

        # Variables
        self.reference_path = None
        self.pdf_paths = []
        self.clf, self.vectorizer = train_classifier()
        self.dbqa_results = []

        # Load Icons
        self.upload_icon = ImageTk.PhotoImage(Image.open("upload.png").resize((24, 24)))
        self.run_icon = ImageTk.PhotoImage(Image.open("run.png").resize((24, 24)))
        self.reset_icon = ImageTk.PhotoImage(Image.open("reset.png").resize((24, 24)))

        # Header Frame
        header_frame = tk.Frame(root, bg="#2c3e50", relief="flat")
        header_frame.pack(fill="x")
        tk.Label(header_frame, text="AI Document Analyzer Pro", font=("Helvetica", 28, "bold"), 
                 bg="#2c3e50", fg="#f1c40f").pack(pady=15)

        # Main Frame
        main_frame = tk.Frame(root, bg="#ffffff", relief="ridge", borderwidth=1)
        main_frame.pack(pady=20, padx=20, fill="both", expand=True)

        # Status Frame
        status_frame = tk.Frame(main_frame, bg="#ecf0f1", relief="groove", borderwidth=2)
        status_frame.pack(fill="x", pady=10)
        self.progress_label = tk.Label(status_frame, text="Status: Ready", font=("Helvetica", 12), 
                                      bg="#ecf0f1", fg="#34495e")
        self.progress_label.pack(side="left", padx=10)
        self.pdf_counter = tk.Label(status_frame, text="PDFs Selected: 0", font=("Helvetica", 12), 
                                   bg="#ecf0f1", fg="#16a085")
        self.pdf_counter.pack(side="right", padx=10)

        # Notebook for Tabs
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill="x", pady=10)
        style = ttk.Style()
        style.configure("TNotebook", background="#ffffff")
        style.configure("TNotebook.Tab", font=("Helvetica", 12, "bold"), padding=[15, 5], 
                        background="#bdc3c7", foreground="#2c3e50")
        style.map("TNotebook.Tab", background=[("selected", "#2c3e50")], foreground=[("selected", "#f1c40f")])

        # Plagiarism Tab
        plagiarism_tab = tk.Frame(notebook, bg="#ffffff")
        notebook.add(plagiarism_tab, text="Plagiarism Check")
        self.setup_plagiarism_tab(plagiarism_tab)

        # Classification Tab
        classification_tab = tk.Frame(notebook, bg="#ffffff")
        notebook.add(classification_tab, text="Classification")
        self.setup_classification_tab(classification_tab)

        # DBQA Tab
        dbqa_tab = tk.Frame(notebook, bg="#ffffff")
        notebook.add(dbqa_tab, text="Question Answering")
        self.setup_dbqa_tab(dbqa_tab)

        # Results and Chart Frame
        results_frame = tk.Frame(main_frame, bg="#ffffff")
        results_frame.pack(fill="both", expand=True, pady=10)

        # Results Text
        results_label_frame = tk.LabelFrame(results_frame, text="Analysis Results", font=("Helvetica", 14, "bold"), 
                                           bg="#ffffff", fg="#2c3e50", relief="ridge", borderwidth=2)
        results_label_frame.pack(side="left", fill="both", expand=True, padx=5)
        self.results_text = scrolledtext.ScrolledText(results_label_frame, width=70, height=15, 
                                                     font=("Courier", 11), bg="#f9f9f9", fg="#2c3e50", wrap=tk.WORD)
        self.results_text.pack(pady=5, padx=5)

        # Errors Text (Narrower width, reduced height)
        errors_label_frame = tk.LabelFrame(results_frame, text="Error Log", font=("Helvetica", 14, "bold"), 
                                          bg="#ffffff", fg="#e74c3c", relief="ridge", borderwidth=2)
        errors_label_frame.pack(side="left", fill="both", expand=False, padx=5, pady=5)
        self.errors_text = scrolledtext.ScrolledText(errors_label_frame, width=40, height=3, 
                                                    font=("Courier", 11), bg="#f9f9f9", fg="#e74c3c", wrap=tk.WORD)
        self.errors_text.pack(pady=5, padx=5)

        # Chart Canvas (Wider area)
        self.chart_frame = tk.LabelFrame(results_frame, text="Statistics", font=("Helvetica", 14, "bold"), 
                                        bg="#ffffff", fg="#2c3e50", relief="ridge", borderwidth=2)
        self.chart_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)

        # Document Request Entry
        doc_frame = tk.Frame(main_frame, bg="#ffffff")
        doc_frame.pack(fill="x", pady=5)
        tk.Label(doc_frame, text="Get Document for Answer #:", font=("Helvetica", 12), bg="#ffffff").pack(side="left")
        self.doc_entry = tk.Entry(doc_frame, width=5, font=("Helvetica", 12))
        self.doc_entry.pack(side="left", padx=5)
        tk.Button(doc_frame, text="Show Document", command=self.show_document, bg="#2ecc71", fg="white", 
                  font=("Helvetica", 12)).pack(side="left")

    def setup_plagiarism_tab(self, tab):
        tk.Label(tab, text="Plagiarism Detection", font=("Helvetica", 18, "bold"), bg="#ffffff", 
                 fg="#e67e22").pack(pady=10)
        tk.Button(tab, text=" Upload Reference", image=self.upload_icon, compound="left", 
                  command=self.upload_reference, bg="#e67e22", fg="white", font=("Helvetica", 12), 
                  width=200, relief="flat").pack(pady=5)
        tk.Button(tab, text=" Upload PDFs (Max 50)", image=self.upload_icon, compound="left", 
                  command=self.upload_pdfs, bg="#e67e22", fg="white", font=("Helvetica", 12), 
                  width=200, relief="flat").pack(pady=5)
        tk.Button(tab, text=" Reset PDFs", image=self.reset_icon, compound="left", 
                  command=self.reset_pdfs, bg="#e74c3c", fg="white", font=("Helvetica", 12), 
                  width=200, relief="flat").pack(pady=5)
        tk.Button(tab, text=" Run Analysis", image=self.run_icon, compound="left", 
                  command=self.check_plagiarism, bg="#d35400", fg="white", font=("Helvetica", 12, "bold"), 
                  width=200, relief="flat").pack(pady=10)
        tk.Label(tab, text="Compare up to 50 PDFs against a reference document.", 
                 font=("Helvetica", 10, "italic"), bg="#ffffff", fg="#7f8c8d").pack()

    def setup_classification_tab(self, tab):
        tk.Label(tab, text="Document Classification", font=("Helvetica", 18, "bold"), bg="#ffffff", 
                 fg="#3498db").pack(pady=10)
        tk.Button(tab, text=" Upload PDFs (Max 50)", image=self.upload_icon, compound="left", 
                  command=self.upload_pdfs, bg="#3498db", fg="white", font=("Helvetica", 12), 
                  width=200, relief="flat").pack(pady=5)
        tk.Button(tab, text=" Reset PDFs", image=self.reset_icon, compound="left", 
                  command=self.reset_pdfs, bg="#e74c3c", fg="white", font=("Helvetica", 12), 
                  width=200, relief="flat").pack(pady=5)
        tk.Button(tab, text=" Run Analysis", image=self.run_icon, compound="left", 
                  command=self.classify, bg="#2980b9", fg="white", font=("Helvetica", 12, "bold"), 
                  width=200, relief="flat").pack(pady=10)
        tk.Label(tab, text="Classify up to 50 PDFs into financial, healthcare, or legal.", 
                 font=("Helvetica", 10, "italic"), bg="#ffffff", fg="#7f8c8d").pack()

    def setup_dbqa_tab(self, tab):
        tk.Label(tab, text="Question Answering", font=("Helvetica", 18, "bold"), bg="#ffffff", 
                 fg="#2ecc71").pack(pady=10)
        tk.Button(tab, text=" Upload PDFs (Max 50)", image=self.upload_icon, compound="left", 
                  command=self.upload_pdfs, bg="#2ecc71", fg="white", font=("Helvetica", 12), 
                  width=200, relief="flat").pack(pady=5)
        tk.Button(tab, text=" Reset PDFs", image=self.reset_icon, compound="left", 
                  command=self.reset_pdfs, bg="#e74c3c", fg="white", font=("Helvetica", 12), 
                  width=200, relief="flat").pack(pady=5)
        self.question_entry = tk.Entry(tab, width=50, font=("Helvetica", 12), relief="flat", 
                                      bg="#ecf0f1", fg="#2c3e50")
        self.question_entry.pack(pady=5)
        tk.Button(tab, text=" Run Analysis", image=self.run_icon, compound="left", 
                  command=self.get_answers, bg="#27ae60", fg="white", font=("Helvetica", 12, "bold"), 
                  width=200, relief="flat").pack(pady=10)
        tk.Label(tab, text="Ask a question about up to 50 PDFs.", 
                 font=("Helvetica", 10, "italic"), bg="#ffffff", fg="#7f8c8d").pack()

    def update_progress(self, message):
        self.progress_label.config(text=f"Status: {message}")

    def update_pdf_counter(self, count):
        self.pdf_counter.config(text=f"PDFs Selected: {count}")

    def upload_reference(self):
        self.reference_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if self.reference_path:
            self.results_text.insert(tk.END, f"Reference PDF: {self.reference_path}\n")

    def upload_pdfs(self):
        self.pdf_paths = filedialog.askopenfilenames(filetypes=[("PDF files", "*.pdf")])
        if self.pdf_paths:
            count = min(len(self.pdf_paths), 50)
            self.update_pdf_counter(count)
            self.results_text.insert(tk.END, f"\nUploaded {count} PDFs:\n")
            for path in self.pdf_paths[:50]:
                self.results_text.insert(tk.END, f"- {path}\n")

    def reset_pdfs(self):
        self.pdf_paths = []
        self.reference_path = None
        self.update_pdf_counter(0)
        self.results_text.delete(1.0, tk.END)
        self.errors_text.delete(1.0, tk.END)
        self.clear_chart()
        self.results_text.insert(tk.END, "PDFs and reference reset.\n")
        self.update_progress("Ready")

    def check_plagiarism(self):
        if not self.reference_path:
            messagebox.showerror("Error", "Please upload a reference PDF!")
            return
        if not self.pdf_paths:
            messagebox.showerror("Error", "Please upload PDFs to compare!")
            return
        try:
            self.results_text.delete(1.0, tk.END)
            self.errors_text.delete(1.0, tk.END)
            self.clear_chart()
            self.results_text.insert(tk.END, "Starting Plagiarism Check...\n\n")
            results = detect_plagiarism(self.reference_path, self.pdf_paths, self)
            self.results_text.insert(tk.END, "=== Plagiarism Results ===\n", "bold")
            plagiarism_stats = {'0-25%': 0, '26-50%': 0, '51-75%': 0, '76-100%': 0, 'Errors': 0}
            for result in results:
                self.results_text.insert(tk.END, f"• {result}\n")
                if "Error" in result:
                    plagiarism_stats['Errors'] += 1
                else:
                    percentage = float(result.split(": ")[-1].rstrip("%"))
                    if 0 <= percentage <= 25:
                        plagiarism_stats['0-25%'] += 1
                    elif 26 <= percentage <= 50:
                        plagiarism_stats['26-50%'] += 1
                    elif 51 <= percentage <= 75:
                        plagiarism_stats['51-75%'] += 1
                    elif 76 <= percentage <= 100:
                        plagiarism_stats['76-100%'] += 1
            self.show_pie_chart(
                {f"{k}: {v} docs" if v > 0 else k: v for k, v in plagiarism_stats.items()},
                "Plagiarism Distribution (Number of Documents)"
            )
            self.update_progress("Ready")
        except Exception as e:
            messagebox.showerror("Error", f"Failed: {str(e)}")

    def classify(self):
        if not self.pdf_paths:
            messagebox.showerror("Error", "Please upload PDFs to classify!")
            return
        try:
            self.results_text.delete(1.0, tk.END)
            self.errors_text.delete(1.0, tk.END)
            self.clear_chart()
            self.results_text.insert(tk.END, "Starting Classification...\n\n")
            results = classify_documents(self.pdf_paths, self.clf, self.vectorizer, self)
            self.results_text.insert(tk.END, "=== Classification Results ===\n", "bold")
            categories = {'financial': 0, 'healthcare': 0, 'legal': 0, 'errors': 0}
            for result in results:
                self.results_text.insert(tk.END, f"• {result}\n")
                if "Error" in result:
                    categories['errors'] += 1
                else:
                    category = result.split(": ")[-1]
                    categories[category] += 1
            self.show_pie_chart(categories, "Classification Distribution")
            self.update_progress("Ready")
        except Exception as e:
            messagebox.showerror("Error", f"Failed: {str(e)}")

    def get_answers(self):
        if not self.pdf_paths:
            messagebox.showerror("Error", "Please upload PDFs to query!")
            return
        question = self.question_entry.get()
        if not question:
            messagebox.showerror("Error", "Please enter a question!")
            return
        try:
            self.results_text.delete(1.0, tk.END)
            self.errors_text.delete(1.0, tk.END)
            self.clear_chart()
            self.results_text.insert(tk.END, f"User: {question}\n\n")
            results, errors = query_documents(self.pdf_paths, question, self)
            self.dbqa_results = results
            self.results_text.insert(tk.END, "=== Answers ===\n", "bold")
            for i, (path, answer) in enumerate(results, 1):
                self.results_text.insert(tk.END, f"AI [{i}]: {answer}\n\n")
            if errors:
                self.errors_text.insert(tk.END, "=== Errors Encountered ===\n", "bold")
                for error in errors:
                    self.errors_text.insert(tk.END, f"• {error}\n")
            else:
                self.errors_text.insert(tk.END, "No errors encountered.\n")
            self.results_text.insert(tk.END, "Tip: Enter an answer number below to see its document.\n")
            self.update_progress("Ready")
        except Exception as e:
            messagebox.showerror("Error", f"Failed: {str(e)}")

    def show_document(self):
        try:
            index = int(self.doc_entry.get()) - 1
            if 0 <= index < len(self.dbqa_results):
                path, _ = self.dbqa_results[index]
                self.results_text.insert(tk.END, f"\nDocument for Answer [{index + 1}]: {path}\n")
            else:
                messagebox.showerror("Error", "Invalid answer number!")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number!")

    def show_pie_chart(self, data, title):
        self.clear_chart()
        fig, ax = plt.subplots(figsize=(7, 5))  # Keep the size reasonable
        labels = [key for key, value in data.items() if value > 0]
        sizes = [value for value in data.values() if value > 0]
        if not sizes:
            labels = ["No Data"]
            sizes = [1]
            colors = ['#7f8c8d']
        else:
            colors = ['#2ecc71', '#e74c3c', '#3498db', '#f1c40f', '#e67e22'][:len(labels)]
        
        wedges, texts, autotexts = ax.pie(sizes, colors=colors, startangle=90, 
                                          autopct='%1.0f%%', pctdistance=0.85, 
                                          textprops={'fontsize': 8})
        
        # Move legend below the chart
        ax.legend(wedges, labels, title="Categories", loc="center", 
                  bbox_to_anchor=(0.5, -0.1), fontsize=8, ncol=3)  # ncol=3 for horizontal layout
        
        for autotext in autotexts:
            autotext.set_fontsize(8)
        ax.axis('equal')
        ax.set_title(title, fontsize=12, pad=10)
        
        # Adjust layout to make room for the legend at the bottom (fixed 'custom' to 'left')
        plt.subplots_adjust(left=0.1, top=0.9, bottom=0.2)  # Leave space at the bottom
        
        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        plt.close(fig)

    def clear_chart(self):
        for widget in self.chart_frame.winfo_children():
            widget.destroy()

    # Tag for bold text
    def configure_tags(self):
        self.results_text.tag_configure("bold", font=("Courier", 11, "bold"))
        self.errors_text.tag_configure("bold", font=("Courier", 11, "bold"))

# Run the GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = DocumentAnalysisGUI(root)
    app.configure_tags()
    root.mainloop()
