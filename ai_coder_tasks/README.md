# AI Coder Task Board

Welcome!  This directory is a **light-weight kanban board** that both you (the human) and the LLM can reference while working on the project.

•  **Each task is a single markdown file.**  Put it in `ai_coder_tasks/tasks/` and name it `<ticket-id>_<slug>.md`  
•  **Keep context close to the task.**  Anything the LLM needs (API routes, example JSON, links to code) lives inside the task file so you don't have to scroll back through chat history.

### Directory layout
```
ai_coder_tasks/
├── README.md               – this file
├── overview.md             – high-level architecture & ring plan
└── tasks/
    ├── _template.md        – copy-paste this for a new ticket
    └── …                   – one file per open task
```

### Workflow
1. **Create or open** a task file when you are ready to work on it.
2. **Attach the file** (or paste its contents) into the chat so the LLM sees the full context.
3. After the task is complete mark `Status:` **Done** (or delete the file).

That's all – minimal friction, maximum shared context.  Happy building!  🎉 