from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool
import streamlit as st

from dotenv import load_dotenv
load_dotenv()

# Streamlit page config
st.set_page_config(page_title="Agentic AI # CrewAI", layout="wide")

# Title and Description
st.title("Content Researcher & Writer - powered by CrewAI")
st.markdown("Generate detailed research reports and engaging blog posts with autonomous AI agents working together as a team.")

# Sidebar
with st.sidebar:
  st.header("Content Settings")

  # Make the text input take up more space
  topic = st.text_area(
    "Enter your topic",
    height=100,
    placeholder="Enter the topic"
  )

  # Adding more sidebar controls (if needed)
  st.markdown("### LLM Settings")
  temperature = st.slider("Model Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)

  # Adding more spacing
  st.markdown("___")

  # Making the generate button more prominent in the sidebar
  generate_button = st.button("Generate Content", type="primary", use_container_width=True)

  # Adding some helpful information
  with st.expander("How to use ?"):
    st.markdown("""
    1. Enter a topic you want research on 
    2. Adjust the model parameters if needed
    3. Click on generate button to start the workflow
    4. Wait for the workflow to complete
    5. View the results in the main page
    6. Download the research report from the bottom of the page (as MD file)
    """)

# Generate Function
def generate_content(topic):
  '''Tool #1'''
  # Initialize LLM
  # We are using OpenAI's gpt-4 model
  llm = LLM(model="gpt-4")

  '''Tool #2'''
  # Initialize Search Tool
  # We are using SerperDevTool to search the web
  # n=10 means we want to search for the top 10 results
  search_tool = SerperDevTool(n=10)

  '''Agent #1'''
  senior_research_analyst = Agent(
    role = "Senior Research Analyst",
    goal = f"Research, Analyze, and Synthesize comprehensive information on {topic} from reliable web sources",
    backstory = "You're an expert research analyst with advanced web research skills. "
                "You excel at finding, analyzing, and synthesizing information from "
                "across the internet using search tools. You're skilled at "
                "distinguishing reliable sources from unreliable ones, "
                "fact-checking, cross-referencing information, and "
                "identifying key patterns and insights. You provide "
                "well-organized research briefs with proper citations "
                "and source verification. Your analysis includes both "
                "raw data and interpreted insights, making complex "
                "information accessible and actionable.", # Backstory -> Making the agent understand its role and responsibilities (it should be detailed and comprehensive)
    allow_delegation = False, # Whether the agent can delegate tasks to other agents
    verbose = True, # Whether the agent should print its thoughts and actions (on terminal)
    tools = [search_tool], # Tools the agent can use
    llm=llm # LLM the agent should use
  )

  '''Agent #2'''
  content_writer = Agent(
    role = "Content Writer",
    goal = "Transform research findings into engaging blog post while maintaining factual accuracy",
    backstory = "You're a skilled content writer specialized in creating "
                "engaging, accessible content from technical research."
                "You work closely with the Senior Research Analyst and excel at maintaining the perfect "
                "balance between informative and entertaining writing, "
                "while ensuring all facts and citations from the research"
                "are properly incorporated. You have a talent for making "
                "complex topics approachable without oversimplifying them.",
      allow_delegation = False,
      verbose = True,
      llm=llm
  )

  '''Tasks'''
  # Task-1 : Research Task
  research_tasks = Task(
      description = ("""
            1. Conduct comprehensive research on {topic} including:
              - Recent developments and news
              - Key industry trends and innovations
              - Expert opinions and analyses
              - Statistical data and market insights
            2. Evaluate source credibility and fact-check all information
            3. Organize findings into a structured research brief
            4. Include all relevant citations and sources
          """), # The description for the task
      expected_output = """A detailed research report containing:
              - Executive summary of key findings
              - Comprehensive analysis of current trends and developments
              - List of verified facts and statistics
              - All citations and links to original sources
              - Clear categorization of main themes and patterns
              Please format with clear sections and bullet points for easy reference.""", # The expected output for the task
      agent = senior_research_analyst, # Assigning the task to the senior_research_analyst agent
  )

  # Task-2 : Content Writing Task
  writer_task = Task(
    description = (
      '''
      Using the research brief provided, create an engaging blog post that:
          1. Transforms technical information into accessible content
          2. Maintains all factual accuracy and citations from the research
          3. Includes:
          - Attention-grabbing introduction
          Well-structured body sections with clear headings
          Compelling conclusion
          4. Preserves all source citations in [Source: URL] format
          5. Includes a References section at the end
      '''), # The description for the task
      expected_output = """A polished blog post in markdown format that:
          - Engages readers while maintaining accuracy
          - Contains properly structured sections
          - Includes Inline citations hyperlinked to the original source url
          - Presents information in an accessible yet informative way
          - Follows proper markdown formatting, use H1 for the title and H3 for the sub-sections""",
      agent = content_writer
  )

  # Defining the crew
  crew = Crew(
    agents = [
      senior_research_analyst, 
      content_writer
    ],
    tasks = [research_tasks, writer_task],
    verbose = True # True means the crew will print its thoughts and actions on the terminal, similar to what we did when defining agents
  )

  return crew.kickoff(inputs = {"topic" : topic}) # Kickoff the crew and start the workflow

# Main content area
if generate_button:
  with st.spinner('Generating content... This may take a moment.'):
    try:
      result = generate_content(topic)
      st.markdown("# Generated Content")
      st.markdown(result)

      # Adding download button
      st.download_button(
        label="Download Content",
        data=result.raw,
        file_name=f" {topic. lower().replace('', '')}_article.md",
        mime="text/markdown"
      )
    except Exception as e:
      st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown("--")
st.markdown("Built with CrewAI, Streamlit and ChatGPT")
