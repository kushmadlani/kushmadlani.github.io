---
title: 'Machine Learning: Starting from Zero'
date: 15-07-2020
---

Just over a year and a half ago I was sitting at a desk trading stocks, knowing close to nothing about Machine Learning and having last coded in a bit of Matlab 4 years earlier at university. This post hopes to give some resources for someone wanting to make a start on their Machine Learning journey, like I did, starting from zero. 

### Learning Python

First up is getting comfortable with Python. Its by far the most used language in Machine Learning being simple, consistent and easily readable. The [Modern Python 3 Bootcamp](https://www.udemy.com/course/the-modern-python3-bootcamp/) on Udemy was outstanding and perhaps the best £16.99 I've spent, but there'll a trove of free courses available elsewhere online.

### Maths

While it might be possible to get by without understanding what is going on underneath the hood of a model or process - to excel and know how, what to use, when (and more importantly why) its necessary to dig into the maths. During my masters course, a grounding in Linear Algebra / Probability / Calculus was essential and it goes without saying the same is true for anyone looking to one day do research in ML and contribute to the field. I was fortunate that I'd covered most of what I needed to know during my undergraduate degree, but here are a couple good places to get started:

* [Mathematics for Machine Learning](https://mml-book.github.io/) (book - availble free online)
* [Mathematics for Machine Learning Specialization](https://www.coursera.org/specializations/mathematics-machine-learning) (online course - Coursera)

### Introductory Courses

Before my course start, I did Andrew Ng's famous [course](https://www.coursera.org/learn/machine-learning) - one of the worlds foremost minds in AI, he does a fantastic job of building up the *intuition* behind some of the core building blocks of ML. The only downside is that the exercises are in Matlab so not as helpful if you're new to coding and just know Python.

A great next step once you've got a bit more comfortable coding is these set of free courses from [fast.ai](http://course18.fast.ai/index.html). They're split into an Introductory course, Deep Learning Part 1 and Deep Learning Part 2. A bonus here is getting comfortable with PyTorch which has overtaken TensorFlow as the most widely used framework for deep learning (at least in terms of papers at top reserach conferences, see chart below.)

![jpg](/images/from_zero/pytorch_tf.png)

### MIT's *Missing Semester*

This [class](https://missing.csail.mit.edu/) gets special mention. Some really clever folks over at MIT decided to create a lecture series on all the little bits that aren't covered in a Computer Science degree, but that are essential and will supercharge your work on a computer. Amongst other things, it got me to start using the command line better, taught me how version control worked and how to debug code well. The course is even more benefital to those without any formal Computer Science background, teaching you things you otherwise wouldn't have picked up through raw expereince during years of assignments/projects.

> "Classes teach you all about advanced topics within CS, from operating systems to machine learning, but there’s one critical subject that’s rarely covered, and is instead left to students to figure out on their own: proficiency with their tools. We’ll teach you how to master the command-line, use a powerful text editor, use fancy features of version control systems, and much more!"

### Git

Version control - managing and tracking changes to source code - is a core part of any software/model development. Git is the most widely used version control system and GitHub is the most popular web-based graphical interface for it. The Missing Semester (above) has a great lecture on the theory behind how version control works but as is the case with many things, the best way to understand how Git/Github works in practice is to use it. I'll admit I struggled to get my head around the workflow at first but many questions to a helpful friend later I got the hang of it (remember that if you're stuck or get an error you don't understand, there's likely a post + solution somehwere on stackoverflow).

### AI ethics

Something that wasn't quite covered in my masters formally was AI ethics/data practices. Armed with such powerful tools that will inevitably shape the world in ways we can't imagine today, its ever more important for an ML practicioner to understand the implications, dangers and limitations of those tools. 

As such, here's some books I learnt a lot from related to the above that have, at the very least, made me aware of some of the broader implications that AI can have on society.

* [Superintelligence: Paths, Dangers, Strategies](https://www.goodreads.com/book/show/20527133-superintelligence) by Nick Bostrom
* [Life 3.0: Being Human in the Age of Artificial Intelligence](https://www.goodreads.com/book/show/34272565-life-3-0?from_search=true&from_srp=true&qid=zLREXESdJR&rank=1) by Max Tegmark

____
### *Final note*

Something that fascinates me about ML is that it doesn't fit into one single discipline - its somewhere inbetween Maths and Computer Science, its both practical and theoretical. It borrows ideas from Physics, Pyschology, Linguistics, Biology (to name a few). Studying ML this past year I've found myself fleeting between coding, writing reports, solving maths problem sheets, designing experiments, reading papers. The Machine Learning researcher's toolbox is vast and ever growing.

ML is moving at such an electric pace and is constantly alight with new research presenting advances in every topic imaginable. I'm only just getting started and it really is an exciting time to be involved in something where there's so much progress. Stay humble about what you do and don't know, keep adapting to the latest techniques, practices and advances and always continue learning.

Here's some topics I'm hoping to learn more about soon:
* Casual Inference
* Gaussian Processes
* Bayesian Deep Learning


*Thanks to my classmates George Lamb, Udeepa Meepangana & Aneesh Pappu for their invaluable discussions about this post #WAKU*