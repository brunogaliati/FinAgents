import yfinance as yf
import pandas as pd
import numpy as np
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from graphviz import Digraph
import os
import matplotlib.pyplot as plt
from datetime import datetime
from fpdf import FPDF
import seaborn as sns
import matplotlib
import re
matplotlib.use('Agg')  # Use non-interactive backend
from dotenv import load_dotenv

load_dotenv()

# OpenAI API key configuration
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# GPT-4o model configuration
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.2)

# Define portfolio details dynamically.
portfolio_details = {
    "AAPL": {"weight": 0.25},
    "MSFT": {"weight": 0.25},
    "GOOGL": {"weight": 0.20},
    "AMZN": {"weight": 0.15},
    "TSLA": {"weight": 0.15}
}

stocks = list(portfolio_details.keys())
current_weights = {stock: info["weight"] for stock, info in portfolio_details.items()}

def get_stock_data(tickers, period="1y"):
    data = yf.download(tickers, period=period, interval="1d")
    if "Adj Close" in data:
        return data["Adj Close"]
    else:
        print("⚠️ Warning: 'Adj Close' not found. Returning closing prices.")
        return data["Close"]

def calculate_portfolio_metrics(stock_data, weights):
    returns = stock_data.pct_change().dropna()
    portfolio_returns = pd.Series(0.0, index=returns.index)
    for stock, weight in weights.items():
        if stock in returns.columns:
            portfolio_returns += returns[stock] * weight
    annual_return = portfolio_returns.mean() * 252
    annual_volatility = portfolio_returns.std() * np.sqrt(252)
    sharpe_ratio = annual_return / annual_volatility
    cumulative_returns = (1 + portfolio_returns).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns / running_max) - 1
    max_drawdown = drawdown.min()
    stock_metrics = {}
    for stock in weights.keys():
        if stock in returns.columns:
            stock_returns = returns[stock]
            stock_metrics[stock] = {
                "annual_return": stock_returns.mean() * 252,
                "annual_volatility": stock_returns.std() * np.sqrt(252),
                "sharpe_ratio": (stock_returns.mean() * 252) / (stock_returns.std() * np.sqrt(252)),
                "beta": stock_returns.cov(portfolio_returns) / portfolio_returns.var()
            }
    return {
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "returns": returns,
        "portfolio_returns": portfolio_returns,
        "stock_metrics": stock_metrics
    }

def validate_and_normalize_allocations(allocation_text):
    """
    Validates and normalizes portfolio allocations to ensure they sum to exactly 100%.
    
    Args:
        allocation_text (str): The allocation recommendations text containing percentages
        
    Returns:
        str: Updated allocation text with normalized percentages
    """
    print("\n=== VALIDATING ALLOCATIONS ===")
    print(f"Original text length: {len(allocation_text)} characters")
    
    # Check if there's a table in the allocation recommendations
    table_start = allocation_text.find("|")
    if table_start != -1:
        print("Found a table in the allocation recommendations")
        
        # Extract the table
        table_lines = []
        in_table = False
        for line in allocation_text[table_start:].split('\n'):
            if line.strip().startswith('|'):
                table_lines.append(line.strip())
                in_table = True
            elif in_table and not line.strip():
                # Empty line after table
                break
        
        if len(table_lines) >= 3:  # At least header, separator, and one data row
            print(f"Found {len(table_lines)} table lines")
            
            # Parse the table to extract allocations
            allocations = {}
            for i in range(2, len(table_lines)):  # Skip header and separator
                cells = [cell.strip() for cell in table_lines[i].split('|')[1:-1]]
                if len(cells) >= 3:
                    asset = cells[0].strip()
                    if asset and "%" in cells[2]:
                        percentage_str = cells[2].replace("%", "").strip()
                        try:
                            percentage = float(percentage_str)
                            allocations[asset] = percentage
                            print(f"  Found in table: {asset} = {percentage}%")
                        except ValueError:
                            print(f"  Error parsing percentage: {percentage_str}")
            
            # Calculate total allocation
            total_allocation = sum(allocations.values())
            print(f"\nOriginal total allocation from table: {total_allocation}%")
            print("Original allocations from table:", allocations)
            
            if abs(total_allocation - 100) > 0.01:  # Allow for small floating point differences
                print(f"\nNeed to normalize allocations to 100% (current: {total_allocation}%)")
                # Normalize allocations to sum to 100%
                scaling_factor = 100 / total_allocation
                print(f"Scaling factor: {scaling_factor}")
                
                # Update allocations
                normalized_allocations = {}
                for asset in allocations:
                    # Scale and round to nearest integer or one decimal place
                    normalized_allocations[asset] = round(allocations[asset] * scaling_factor * 10) / 10
                
                # Ensure the sum is exactly 100% after rounding
                total_after_scaling = sum(normalized_allocations.values())
                if abs(total_after_scaling - 100) > 0.01:
                    # Adjust the largest allocation to make the sum exactly 100%
                    largest_asset = max(normalized_allocations.items(), key=lambda x: x[1])[0]
                    adjustment = round((100 - total_after_scaling) * 10) / 10
                    print(f"Adjusting largest asset ({largest_asset}) by {adjustment}% to make total exactly 100%")
                    normalized_allocations[largest_asset] += adjustment
                
                print(f"Normalized total allocation: {sum(normalized_allocations.values())}%")
                print("Normalized allocations:", normalized_allocations)
                
                # Update the table with normalized percentages
                print("\nUpdating table with normalized percentages...")
                updated_table_lines = table_lines[:2]  # Keep header and separator
                
                for i in range(2, len(table_lines)):
                    line = table_lines[i]
                    cells = [cell.strip() for cell in line.split('|')[1:-1]]
                    if len(cells) >= 3:
                        asset = cells[0].strip()
                        if asset in normalized_allocations:
                            percentage = normalized_allocations[asset]
                            percentage_str = f"{int(percentage)}%" if percentage.is_integer() else f"{percentage:.1f}%"
                            cells[2] = percentage_str
                            updated_line = "| " + " | ".join(cells) + " |"
                            print(f"  Updated: {updated_line}")
                            updated_table_lines.append(updated_line)
                        else:
                            updated_table_lines.append(line)
                    else:
                        updated_table_lines.append(line)
                
                # Replace the table in the allocation text
                updated_table = "\n".join(updated_table_lines)
                allocation_text = allocation_text[:table_start] + updated_table + allocation_text[table_start + len("\n".join(table_lines)):]
                
                print("Table updated successfully")
                return allocation_text
    
    # If no table found or table processing failed, use the regex approach
    # Extract all percentage allocations using regex
    # Main pattern for allocations like "AAPL: 15%" or "AGG (Bonds): 15%"
    percentage_pattern = r'(\w+(?:\s+\([^)]+\))?)\s*:\s*(\d+(?:\.\d+)?)%'
    
    # Pattern for nested allocations like "JNJ 3%" within category sections
    nested_percentage_pattern = r'(\w+)\s+(\d+(?:\.\d+)?)%'
    
    # Pattern for markdown table rows like "| AAPL | 20% | 15% |"
    table_row_pattern = r'\|\s*([^|]+)\s*\|\s*\d+(?:\.\d+)?%\s*\|\s*(\d+(?:\.\d+)?)%\s*\|'
    
    allocations = {}
    
    # List of phrases to exclude from being considered as assets
    excluded_phrases = ["exactly", "precisely", "total", "sum", "allocation"]
    
    # Find all main allocations (e.g., "AAPL: 15%")
    print("\nSearching for main allocations (e.g., 'AAPL: 15%'):")
    for match in re.finditer(percentage_pattern, allocation_text):
        asset, percentage = match.groups()
        asset = asset.strip()
        
        # Skip excluded phrases
        if any(phrase in asset.lower() for phrase in excluded_phrases):
            print(f"  Skipping excluded phrase: {asset}")
            continue
            
        print(f"  Found: {asset} = {percentage}%")
        allocations[asset] = float(percentage)
    
    # Find all nested allocations (e.g., "JNJ 3%")
    print("\nSearching for nested allocations (e.g., 'JNJ 3%'):")
    for match in re.finditer(nested_percentage_pattern, allocation_text):
        asset, percentage = match.groups()
        asset = asset.strip()
        
        # Skip excluded phrases
        if any(phrase in asset.lower() for phrase in excluded_phrases):
            print(f"  Skipping excluded phrase: {asset}")
            continue
            
        if asset not in allocations:  # Avoid duplicates
            print(f"  Found: {asset} = {percentage}%")
            allocations[asset] = float(percentage)
    
    # Find all table row allocations
    print("\nSearching for table row allocations (e.g., '| AAPL | ... | 15% |'):")
    for match in re.finditer(table_row_pattern, allocation_text):
        asset, percentage = match.groups()
        asset = asset.strip()
        
        # Skip excluded phrases
        if any(phrase in asset.lower() for phrase in excluded_phrases):
            print(f"  Skipping excluded phrase: {asset}")
            continue
            
        if asset not in allocations:  # Avoid duplicates
            print(f"  Found: {asset} = {percentage}%")
            allocations[asset] = float(percentage)
    
    # Calculate total allocation
    total_allocation = sum(allocations.values())
    print(f"\nOriginal total allocation: {total_allocation}%")
    print("Original allocations:", allocations)
    
    if abs(total_allocation - 100) > 0.01:  # Allow for small floating point differences
        print(f"\nNeed to normalize allocations to 100% (current: {total_allocation}%)")
        # Normalize allocations to sum to 100%
        scaling_factor = 100 / total_allocation
        print(f"Scaling factor: {scaling_factor}")
        
        # Update allocations
        normalized_allocations = {}
        for asset in allocations:
            # Scale and round to nearest integer or one decimal place
            normalized_allocations[asset] = round(allocations[asset] * scaling_factor * 10) / 10
        
        # Ensure the sum is exactly 100% after rounding
        total_after_scaling = sum(normalized_allocations.values())
        if abs(total_after_scaling - 100) > 0.01:
            # Adjust the largest allocation to make the sum exactly 100%
            largest_asset = max(normalized_allocations.items(), key=lambda x: x[1])[0]
            adjustment = round((100 - total_after_scaling) * 10) / 10
            print(f"Adjusting largest asset ({largest_asset}) by {adjustment}% to make total exactly 100%")
            normalized_allocations[largest_asset] += adjustment
        
        print(f"Normalized total allocation: {sum(normalized_allocations.values())}%")
        print("Normalized allocations:", normalized_allocations)
        
        # Update the allocation text with normalized percentages
        print("\nUpdating allocation text with normalized percentages...")
        
        # Replace main allocations
        for asset, percentage in normalized_allocations.items():
            # Format percentage with one decimal place if it's not a whole number
            percentage_str = f"{int(percentage)}%" if percentage.is_integer() else f"{percentage:.1f}%"
            
            # Replace in main pattern
            pattern = rf'{re.escape(asset)}\s*:\s*\d+(?:\.\d+)?%'
            if re.search(pattern, allocation_text):
                print(f"  Replacing main allocation: {asset}: {percentage_str}")
                allocation_text = re.sub(
                    pattern, 
                    f'{asset}: {percentage_str}', 
                    allocation_text
                )
            
            # Replace in nested pattern
            pattern = rf'{re.escape(asset)}\s+\d+(?:\.\d+)?%'
            if re.search(pattern, allocation_text):
                print(f"  Replacing nested allocation: {asset} {percentage_str}")
                allocation_text = re.sub(
                    pattern, 
                    f'{asset} {percentage_str}', 
                    allocation_text
                )
            
            # Replace in table rows
            pattern = rf'\|\s*{re.escape(asset)}\s*\|\s*\d+(?:\.\d+)?%\s*\|\s*\d+(?:\.\d+)?%\s*\|'
            if re.search(pattern, allocation_text):
                print(f"  Replacing table row: | {asset} | x% | {percentage_str} |")
                # Preserve the middle column content
                def replace_table_row(match):
                    full_match = match.group(0)
                    parts = full_match.split('|')
                    if len(parts) >= 5:
                        parts[0] = '|'
                        parts[1] = f' {asset} '
                        # Keep parts[2] (current allocation) unchanged
                        parts[3] = f' {percentage_str} '
                        parts[4] = '|'
                        return '|'.join(parts[:5])
                    return full_match
                
                allocation_text = re.sub(pattern, replace_table_row, allocation_text)
    
    print("\n=== VALIDATION COMPLETE ===")
    return allocation_text

def generate_charts(stock_data, portfolio_metrics, chart_dir="charts"):
    os.makedirs(chart_dir, exist_ok=True)
    chart_paths = {}
    # Normalized Stock Performance
    plt.figure(figsize=(10, 6))
    normalized_prices = stock_data / stock_data.iloc[0]
    normalized_prices.plot(title="Normalized Stock Performance")
    plt.ylabel("Normalized Price")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    performance_path = os.path.join(chart_dir, "normalized_performance.png")
    plt.savefig(performance_path, dpi=300)
    plt.close()
    chart_paths["performance"] = performance_path

    # Portfolio Returns Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(portfolio_metrics["portfolio_returns"], kde=True, bins=50)
    plt.title("Portfolio Daily Returns Distribution")
    plt.xlabel("Daily Return")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    returns_path = os.path.join(chart_dir, "returns_distribution.png")
    plt.savefig(returns_path, dpi=300)
    plt.close()
    chart_paths["returns"] = returns_path

    # Correlation Matrix
    plt.figure(figsize=(10, 8))
    correlation = portfolio_metrics["returns"].corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title("Stock Correlation Matrix")
    plt.tight_layout()
    correlation_path = os.path.join(chart_dir, "correlation_matrix.png")
    plt.savefig(correlation_path, dpi=300)
    plt.close()
    chart_paths["correlation"] = correlation_path

    # Cumulative Returns
    plt.figure(figsize=(10, 6))
    cumulative_returns = (1 + portfolio_metrics["portfolio_returns"]).cumprod()
    cumulative_returns.plot(title="Portfolio Cumulative Returns")
    plt.ylabel("Cumulative Return")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    cumulative_path = os.path.join(chart_dir, "cumulative_returns.png")
    plt.savefig(cumulative_path, dpi=300)
    plt.close()
    chart_paths["cumulative"] = cumulative_path

    # Risk-Return Scatter Plot
    plt.figure(figsize=(10, 6))
    stock_metrics = portfolio_metrics["stock_metrics"]
    returns_list = [stock_metrics[stock]["annual_return"] * 100 for stock in stocks]
    volatilities = [stock_metrics[stock]["annual_volatility"] * 100 for stock in stocks]
    plt.scatter(volatilities, returns_list, s=100)
    for i, stock in enumerate(stocks):
        plt.annotate(stock, (volatilities[i], returns_list[i]), textcoords="offset points", xytext=(5,5), ha='left')
    plt.title("Risk-Return Profile")
    plt.xlabel("Annual Volatility (%)")
    plt.ylabel("Annual Return (%)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    risk_return_path = os.path.join(chart_dir, "risk_return.png")
    plt.savefig(risk_return_path, dpi=300)
    plt.close()
    chart_paths["risk_return"] = risk_return_path

    # Current Portfolio Allocation Pie Chart
    plt.figure(figsize=(10, 6))
    plt.pie(list(current_weights.values()), labels=list(current_weights.keys()), autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title("Current Portfolio Allocation")
    plt.tight_layout()
    allocation_path = os.path.join(chart_dir, "current_allocation.png")
    plt.savefig(allocation_path, dpi=300)
    plt.close()
    chart_paths["allocation"] = allocation_path

    print(f"Charts saved to {chart_dir} directory")
    return chart_paths

class PortfolioPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.WIDTH = 210
        self.HEIGHT = 297
        
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(self.WIDTH - 20, 10, 'Investment Portfolio Analysis', 0, 1, 'R')
        self.set_font('Arial', 'I', 10)
        self.cell(self.WIDTH - 20, 10, f'Generated on {datetime.now().strftime("%Y-%m-%d")}', 0, 1, 'R')
        self.line(10, 30, self.WIDTH - 10, 30)
        self.ln(20)
        
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
        
    def chapter_title(self, title):
        self.set_font('Arial', 'B', 14)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 10, title, 0, 1, 'L', 1)
        self.ln(5)
        
    def chapter_body(self, body):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 6, body)
        self.ln(5)
        
    def add_image(self, image_path, w=190):
        self.image(image_path, x=10, w=w)
        self.ln(5)
        
    def add_table(self, headers, data, col_widths=None):
        # Calculate column widths if not provided
        if col_widths is None:
            # Ensure table fits within page margins (190mm width)
            available_width = 190
            col_widths = [available_width / len(headers)] * len(headers)
            
            # Adjust column widths based on content
            if len(headers) > 3:
                # For tables with many columns, make first column wider for readability
                col_widths[0] = available_width * 0.25  # 25% for first column
                remaining_width = available_width * 0.75  # 75% for remaining columns
                for i in range(1, len(headers)):
                    col_widths[i] = remaining_width / (len(headers) - 1)
        
        line_height = 7
        self.set_font('Arial', 'B', 10)
        
        # Draw header row
        for i, header in enumerate(headers):
            self.cell(col_widths[i], line_height, header, 1, 0, 'C')
        self.ln(line_height)
        
        # Draw data rows
        self.set_font('Arial', '', 10)
        for row in data:
            for i, item in enumerate(row):
                self.cell(col_widths[i], line_height, str(item), 1, 0, 'C')
            self.ln(line_height)
        self.ln(5)

def create_pdf_report(report_content, chart_paths, portfolio_metrics, filename="Portfolio_Report.pdf"):
    # Validate and normalize allocation recommendations
    if "allocation_recommendations" in report_content:
        report_content["allocation_recommendations"] = validate_and_normalize_allocations(
            report_content["allocation_recommendations"]
        )
    
    pdf = PortfolioPDF()
    
    pdf.add_page()
    pdf.chapter_title("Executive Summary")
    pdf.chapter_body(report_content.get("executive_summary", "No Executive Summary provided."))
    
    pdf.add_page()
    pdf.chapter_title("Market Overview")
    pdf.chapter_body(report_content.get("market_overview", "No Market Overview provided."))
    
    pdf.add_page()
    pdf.chapter_title("Portfolio Performance Analysis")
    portfolio_performance = report_content.get("portfolio_performance_analysis", 
                                              report_content.get("portfolio_performance", 
                                                                "No Portfolio Performance Analysis provided."))
    pdf.chapter_body(portfolio_performance)
    if "performance" in chart_paths:
        pdf.add_image(chart_paths["performance"])
    if "cumulative" in chart_paths:
        pdf.add_image(chart_paths["cumulative"])
    
    pdf.chapter_title("Portfolio Metrics")
    headers = ["Metric", "Value"]
    data = [
        ["Annual Return", f"{portfolio_metrics['annual_return']*100:.2f}%"],
        ["Annual Volatility", f"{portfolio_metrics['annual_volatility']*100:.2f}%"],
        ["Sharpe Ratio", f"{portfolio_metrics['sharpe_ratio']:.2f}"],
        ["Maximum Drawdown", f"{portfolio_metrics['max_drawdown']*100:.2f}%"]
    ]
    # Use custom column widths for this table
    col_widths = [95, 95]  # Equal width for both columns
    pdf.add_table(headers, data, col_widths)
    
    pdf.add_page()
    pdf.chapter_title("Risk Assessment")
    pdf.chapter_body(report_content.get("risk_assessment", "No Risk Assessment provided."))
    if "correlation" in chart_paths:
        pdf.add_image(chart_paths["correlation"])
    if "returns" in chart_paths:
        pdf.add_image(chart_paths["returns"])
    if "risk_return" in chart_paths:
        pdf.add_image(chart_paths["risk_return"])
    
    pdf.add_page()
    pdf.chapter_title("Individual Stock Analysis")
    headers = ["Stock", "Weight", "Return", "Volatility", "Sharpe", "Beta"]
    data = []
    for stock in stocks:
        metrics = portfolio_metrics["stock_metrics"].get(stock, {})
        data.append([
            stock,
            f"{current_weights.get(stock, 0)*100:.1f}%",
            f"{metrics.get('annual_return', 0)*100:.2f}%",
            f"{metrics.get('annual_volatility', 0)*100:.2f}%",
            f"{metrics.get('sharpe_ratio', 0):.2f}",
            f"{metrics.get('beta', 0):.2f}"
        ])
    # Use custom column widths for this table to ensure all columns fit
    col_widths = [40, 30, 30, 30, 30, 30]  # Total: 190mm (fits within page margins)
    pdf.add_table(headers, data, col_widths)
    
    pdf.add_page()
    pdf.chapter_title("Allocation Recommendations")
    # Use a smaller font size for this section
    pdf.set_font('Arial', '', 9)  # Decreased from 11 to 9
    allocation_text = report_content.get("allocation_recommendations", "No Allocation Recommendations provided.")
    
    # Check if there's a table in the allocation recommendations
    table_start = allocation_text.find("|")
    if table_start != -1:
        # Split the text before and after the table
        before_table = allocation_text[:table_start].strip()
        
        # Find the end of the table (empty line after table)
        table_lines = []
        after_table = ""
        in_table = False
        for line in allocation_text[table_start:].split('\n'):
            if line.strip().startswith('|'):
                table_lines.append(line.strip())
                in_table = True
            elif in_table and not line.strip():
                # Empty line after table
                in_table = False
                after_table = line
            elif not in_table:
                after_table += line + '\n'
        
        # Add the text before the table
        pdf.multi_cell(0, 5, before_table)  # Decreased line height from 6 to 5
        
        # Parse and add the table
        if len(table_lines) >= 2:  # At least header and separator
            # Parse header
            header_cells = [cell.strip() for cell in table_lines[0].split('|')[1:-1]]
            
            # Skip separator line
            data_rows = []
            for i in range(2, len(table_lines)):
                cells = [cell.strip() for cell in table_lines[i].split('|')[1:-1]]
                if cells:
                    data_rows.append(cells)
            
            # Add the table with dynamic column widths
            num_columns = len(header_cells)
            if num_columns > 0:
                # Calculate column widths based on number of columns
                available_width = 190  # Total available width in mm
                col_widths = [available_width / num_columns] * num_columns
                
                # Adjust first column to be wider if there are multiple columns
                if num_columns > 2:
                    col_widths[0] = available_width * 0.4  # 40% for first column
                    remaining_width = available_width * 0.6  # 60% for remaining columns
                    for i in range(1, num_columns):
                        col_widths[i] = remaining_width / (num_columns - 1)
                
                # Use smaller font for table
                pdf.set_font('Arial', 'B', 8)  # Header font
                line_height = 6  # Decreased from 7
                
                # Draw header row
                for i, header in enumerate(header_cells):
                    pdf.cell(col_widths[i], line_height, header, 1, 0, 'C')
                pdf.ln(line_height)
                
                # Draw data rows
                pdf.set_font('Arial', '', 8)  # Data font
                for row in data_rows:
                    for i, item in enumerate(row):
                        if i < len(col_widths):  # Ensure we don't exceed column widths
                            pdf.cell(col_widths[i], line_height, str(item), 1, 0, 'C')
                    pdf.ln(line_height)
                pdf.ln(3)  # Decreased from 5
        
        # Add the text after the table
        pdf.set_font('Arial', '', 9)  # Reset to section font size
        pdf.multi_cell(0, 5, after_table)  # Decreased line height
    else:
        # No table found, just add the text
        pdf.multi_cell(0, 5, allocation_text)  # Decreased line height
    
    if "allocation" in chart_paths:
        pdf.add_image(chart_paths["allocation"])
    
    # Reset font size for next sections
    pdf.set_font('Arial', '', 11)
    
    pdf.add_page()
    pdf.chapter_title("Implementation Strategy")
    pdf.chapter_body(report_content.get("implementation_strategy", "No Implementation Strategy provided."))
    
    pdf.add_page()
    pdf.chapter_title("Future Outlook")
    pdf.chapter_body(report_content.get("future_outlook", "No Future Outlook provided."))
    
    pdf.add_page()
    pdf.chapter_title("Conclusion")
    pdf.chapter_body(report_content.get("conclusion", "No Conclusion provided."))
    
    pdf.output(filename)
    print(f"PDF report saved as {filename}")
    return filename

# ------------------------------
# Agent Definitions
# ------------------------------
risk_analyst = Agent(
    role="Risk Analyst",
    goal="Evaluate portfolio volatility and risks, suggesting specific changes to improve risk-adjusted returns, including diversification across different asset classes.",
    backstory="You are a financial analyst with expertise in risk assessment, quantitative strategies, and multi-asset portfolio construction. You have deep knowledge of various asset classes including stocks, bonds, REITs, commodities, and alternative investments.",
    verbose=True,
    llm=llm
)

market_analyst = Agent(
    role="Market Analyst",
    goal="Provide a deep analysis of current market conditions and a 12-month outlook for each sector in the portfolio, identifying new sectors and specific companies that could enhance diversification.",
    backstory="You are a seasoned market analyst with expertise in macroeconomic trends, sector analysis, and stock selection. You have a proven track record of identifying emerging sectors and high-potential companies across global markets.",
    verbose=True,
    llm=llm
)

allocation_optimizer = Agent(
    role="Allocation Optimizer",
    goal="Propose specific changes to portfolio allocation to maximize risk-adjusted returns, including new asset classes and specific investment vehicles.",
    backstory="You are an expert in portfolio optimization and quantitative finance using modern portfolio theory. You specialize in multi-asset allocation strategies and have extensive knowledge of ETFs, mutual funds, and individual securities across global markets.",
    verbose=True,
    llm=llm
)

portfolio_manager = Agent(
    role="Portfolio Manager",
    goal="Make final allocation decisions based on risk analysis and optimization suggestions, providing specific implementation steps and investment vehicles.",
    backstory="You are an experienced portfolio manager responsible for strategic investment decisions. You have deep expertise in asset allocation, security selection, and portfolio implementation across various market conditions. You provide actionable advice with specific investment recommendations.",
    verbose=True,
    llm=llm
)

report_generator = Agent(
    role="Investment Report Writer",
    goal="Create a comprehensive investment report for the client that includes all required sections with specific, actionable recommendations.",
    backstory="You are a skilled financial writer who translates complex analysis into clear, detailed reports for high-net-worth clients. Your reports include specific investment recommendations, including ticker symbols, allocation percentages, and implementation steps. Your report must include the following sections with standardized headers:\n\n"
             "EXECUTIVE SUMMARY\nMARKET OVERVIEW\nPORTFOLIO PERFORMANCE ANALYSIS\nRISK ASSESSMENT\nALLOCATION RECOMMENDATIONS\nIMPLEMENTATION STRATEGY\nFUTURE OUTLOOK\nCONCLUSION",
    verbose=True,
    llm=llm
)

# ------------------------------
# Task Definitions
# ------------------------------
stock_data = get_stock_data(stocks)
pm_metrics = calculate_portfolio_metrics(stock_data, current_weights)
recent_data = stock_data.tail(30)
recent_data_str = recent_data.to_csv(index=True)

metrics_summary = f"""
Portfolio Metrics:
- Annual Return: {pm_metrics['annual_return']:.4f} ({pm_metrics['annual_return']*100:.2f}%)
- Annual Volatility: {pm_metrics['annual_volatility']:.4f} ({pm_metrics['annual_volatility']*100:.2f}%)
- Sharpe Ratio: {pm_metrics['sharpe_ratio']:.4f}
- Maximum Drawdown: {pm_metrics['max_drawdown']:.4f} ({pm_metrics['max_drawdown']*100:.2f}%)
"""

stock_metrics_str = "Individual Stock Metrics:\n"
for stock, metrics in pm_metrics["stock_metrics"].items():
    stock_metrics_str += f"\n{stock}:\n"
    stock_metrics_str += f"- Annual Return: {metrics['annual_return']*100:.2f}%\n"
    stock_metrics_str += f"- Annual Volatility: {metrics['annual_volatility']*100:.2f}%\n"
    stock_metrics_str += f"- Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
    stock_metrics_str += f"- Beta: {metrics['beta']:.2f}\n"

risk_analysis_task = Task(
    description=f"""Analyze the portfolio risks based on recent data and calculated metrics.

Recent stock data:
{recent_data_str}

{metrics_summary}

{stock_metrics_str}

Current portfolio weights:
{current_weights}

Identify volatility patterns, correlation risks, and potential market risks.
Suggest risk mitigation strategies with detailed reasoning.

Go beyond basic rebalancing and provide specific recommendations for:
1. Asset class diversification (bonds, REITs, commodities, etc.) with specific ETFs or securities
2. Geographic diversification with specific international market exposure recommendations
3. Factor-based diversification strategies (value, growth, quality, etc.)
4. Hedging strategies with specific implementation methods

For each recommendation, provide specific investment vehicles (with ticker symbols where applicable) and suggested allocation percentages.
""",
    agent=risk_analyst,
    expected_output="Detailed risk analysis report with specific diversification and risk mitigation recommendations."
)

market_analysis_task = Task(
    description=f"""Provide a deep analysis of current market conditions and a 12-month outlook for each sector represented in the portfolio.

The portfolio contains the following stocks:
- AAPL (Apple Inc.): {portfolio_details["AAPL"]["weight"] * 100}%
- MSFT (Microsoft Corp.): {portfolio_details["MSFT"]["weight"] * 100}%
- GOOGL (Alphabet Inc.): {portfolio_details["GOOGL"]["weight"] * 100}%
- AMZN (Amazon.com Inc.): {portfolio_details["AMZN"]["weight"] * 100}%
- TSLA (Tesla Inc.): {portfolio_details["TSLA"]["weight"] * 100}%

First, classify each stock into appropriate sectors and industries based on their business activities.

Then, for each identified sector, please:
1. Analyze current conditions and trends.
2. Identify growth opportunities and risks.
3. Provide a 12-month outlook with key factors to watch.
4. Suggest how these trends might impact each stock within the sector.

Additionally, identify 3-5 new sectors not currently represented in the portfolio that offer diversification benefits and growth potential.
For each new sector, recommend 2-3 specific companies (with ticker symbols) that are leaders or emerging players.

Your analysis should be data-driven, forward-looking, and include specific investment recommendations.
""",
    agent=market_analyst,
    expected_output="Comprehensive market analysis with specific sector and company recommendations for diversification."
)

allocation_task = Task(
    description=f"""Based on the risk analysis and market outlook, suggest adjustments to the portfolio allocation.

{metrics_summary}

{stock_metrics_str}

Current portfolio weights:
{current_weights}

Provide specific weight recommendations for each existing stock, ensuring balance between risk and return.
Additionally, recommend:

1. New asset classes to add to the portfolio (bonds, REITs, commodities, etc.) with specific ETFs or securities and allocation percentages
2. Specific new stocks from underrepresented sectors with suggested allocation percentages
3. Geographic diversification with international market exposure recommendations
4. Factor-based allocation strategies (value, growth, quality, etc.)

For each recommendation, provide specific investment vehicles (with ticker symbols) and suggested allocation percentages.
Justify your recommendations using quantitative and qualitative reasoning.

IMPORTANT: All allocation percentages MUST sum to EXACTLY 100%. Double-check your math to ensure the total allocation is precisely 100%.
""",
    agent=allocation_optimizer,
    expected_output="Comprehensive portfolio allocation proposal with specific investment recommendations across asset classes and sectors."
)

manager_task = Task(
    description=f"""Review the suggestions from the risk analyst, market analyst, and allocation optimizer.
Decide on the final allocation while considering transaction costs and tax implications.
Provide a detailed implementation strategy with specific steps and investment vehicles.

Your final recommendations should include:
1. Specific allocation percentages for all recommended investments (existing and new)
2. Specific ETFs, mutual funds, or individual securities for each asset class and sector (with ticker symbols)
3. Implementation priority and timeline
4. Tax-efficient implementation strategies
5. Ongoing monitoring recommendations

IMPORTANT: All allocation percentages MUST sum to EXACTLY 100%. Double-check your math to ensure the total allocation is precisely 100%.

Be specific and actionable in your recommendations, providing a clear roadmap for portfolio implementation.
""",
    agent=portfolio_manager,
    expected_output="Final portfolio allocation decision with detailed implementation strategy and specific investment recommendations."
)

report_task = Task(
    description=f"""Create a comprehensive investment report for the client based on all previous analyses.
The report must include the following sections with the exact headers:
    
EXECUTIVE SUMMARY
Provide a summary of key findings and recommendations, including the most important specific investment actions to take.

MARKET OVERVIEW
Detail current market conditions and sector insights, including specific sectors to increase or decrease exposure to.

PORTFOLIO PERFORMANCE ANALYSIS
Include detailed analysis of portfolio metrics and historical trends, with specific performance drivers and detractors.

RISK ASSESSMENT
Discuss identified risks and proposed mitigation strategies, including specific diversification recommendations across asset classes.

ALLOCATION RECOMMENDATIONS
Present specific portfolio rebalancing suggestions with exact allocation percentages and ticker symbols for all recommended investments.
Include a clear table showing current vs. recommended allocations for all assets.
Ensure recommendations include diversification across:
- Asset classes (stocks, bonds, alternatives, etc.)
- Sectors (both existing and new sectors)
- Geographic regions
- Investment styles/factors

IMPORTANT: All allocation percentages MUST sum to EXACTLY 100%. Double-check your math to ensure the total allocation is precisely 100%.

IMPLEMENTATION STRATEGY
Outline a step-by-step plan for executing the recommendations, including:
- Specific securities to buy and sell with ticker symbols
- Implementation timeline and priority
- Tax considerations
- Cost-efficient implementation methods

FUTURE OUTLOOK
Provide projections and a 12-month outlook, including specific market catalysts and risks to monitor.

CONCLUSION
Summarize the final recommendations and key takeaways, emphasizing the most important actions to take.

Ensure that every section is complete, specific, and actionable with concrete investment recommendations.
""",
    agent=report_generator,
    expected_output="A professional investment report in plain text with all required sections complete and specific actionable recommendations."
)

# ------------------------------
# Create multi-agent system (without validator/fixer)
# ------------------------------
crew = Crew(
    agents=[
        risk_analyst, market_analyst, allocation_optimizer,
        portfolio_manager, report_generator
    ],
    tasks=[
        risk_analysis_task, market_analysis_task, allocation_task,
        manager_task, report_task
    ],
    verbose=True
)

def generate_workflow_diagram():
    try:
        # Create a new Digraph with improved styling
        dot = Digraph(comment='Portfolio Analysis Workflow', 
                      format='png',
                      engine='dot')
        
        # Set graph attributes for better appearance
        dot.attr(rankdir='TB',  # Top to bottom layout
                 size='8,5',    # Size in inches
                 dpi='300',     # Higher resolution
                 bgcolor='#f7f7f7',  # Light background
                 fontname='Arial',
                 fontsize='14',
                 margin='0.5,0.5')
        
        # Set default node attributes
        dot.attr('node', 
                 shape='ellipse',
                 style='filled,rounded',
                 color='#333333',
                 fontname='Arial',
                 fontsize='14',
                 height='1.2',
                 width='2.5',
                 penwidth='2')
        
        # Set default edge attributes
        dot.attr('edge', 
                 fontname='Arial',
                 fontsize='12',
                 fontcolor='#505050',
                 color='#666666',
                 penwidth='1.5',
                 arrowsize='0.8')
        
        # Create nodes with custom colors for each agent - using proper HTML labels
        # Note: We need to set HTML=True to enable HTML-like labels
        dot.node("A", label='<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0"><TR><TD><FONT POINT-SIZE="16"><B>📊 Risk Analyst</B></FONT></TD></TR></TABLE>>', 
                 fillcolor='#E6F3FF', fontcolor='#0066CC', style='filled,rounded', shape='ellipse', _attributes={"fontname": "Arial"})
        
        dot.node("B", label='<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0"><TR><TD><FONT POINT-SIZE="16"><B>🌎 Market Analyst</B></FONT></TD></TR></TABLE>>', 
                 fillcolor='#E6FFE6', fontcolor='#006600', style='filled,rounded', shape='ellipse', _attributes={"fontname": "Arial"})
        
        dot.node("C", label='<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0"><TR><TD><FONT POINT-SIZE="16"><B>📈 Allocation Optimizer</B></FONT></TD></TR></TABLE>>', 
                 fillcolor='#FFF0E6', fontcolor='#CC6600', style='filled,rounded', shape='ellipse', _attributes={"fontname": "Arial"})
        
        dot.node("D", label='<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0"><TR><TD><FONT POINT-SIZE="16"><B>📝 Portfolio Manager</B></FONT></TD></TR></TABLE>>', 
                 fillcolor='#F3E6FF', fontcolor='#660099', style='filled,rounded', shape='ellipse', _attributes={"fontname": "Arial"})
        
        dot.node("E", label='<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0"><TR><TD><FONT POINT-SIZE="16"><B>📄 Report Generator</B></FONT></TD></TR></TABLE>>', 
                 fillcolor='#FFE6E6', fontcolor='#CC0000', style='filled,rounded', shape='ellipse', _attributes={"fontname": "Arial"})
        
        # Create edges with meaningful connections
        dot.edge("A", "C", label="Risk analysis")
        dot.edge("B", "C", label="Market outlook")
        dot.edge("C", "D", label="Allocation proposal")
        dot.edge("D", "E", label="Final allocation decision")
        dot.edge("A", "E", label="Risk assessment")
        dot.edge("B", "E", label="Market insights")
        
        # Render the graph
        dot.render("portfolio_workflow", format="png", cleanup=True)
        print("Enhanced diagram generated: portfolio_workflow.png")
    except Exception as e:
        print("Error generating workflow diagram:", e)
        print("Make sure Graphviz is properly installed.")

# ------------------------------
# Execute the multi-agent system
# ------------------------------
# Execute the multi-agent system
current_date = datetime.now().strftime("%Y-%m-%d")
try:
    print(f"\n{'='*50}\nStarting Portfolio Analysis Crew\n{'='*50}\n")
    result = crew.kickoff()
    print("\n\nFINAL REPORT:")
    print(result)
    
    # Parse the report text into sections using str(result)
    report_sections = {}
    current_section = None
    content = []
    
    # Convert result to string and split by lines
    report_text = str(result)
    report_lines = report_text.split('\n')
    
    # Define section headers to look for
    section_headers = [
        "EXECUTIVE SUMMARY", "MARKET OVERVIEW", "PORTFOLIO PERFORMANCE ANALYSIS",
        "RISK ASSESSMENT", "ALLOCATION RECOMMENDATIONS", "IMPLEMENTATION STRATEGY",
        "FUTURE OUTLOOK", "CONCLUSION"
    ]
    
    # Process each line to extract sections
    for line in report_lines:
        line_clean = line.strip()
        # Check if this line is a section header
        is_header = False
        for header in section_headers:
            if header in line_clean.upper():
                is_header = True
                if current_section:
                    report_sections[current_section.lower().replace(' ', '_')] = "\n".join(content)
                current_section = header
                content = []
                break
        
        # If not a header and we have a current section, add to content
        if not is_header and current_section:
            content.append(line)
    
    # Add the last section if there is one
    if current_section and content:
        report_sections[current_section.lower().replace(' ', '_')] = "\n".join(content)
    
    # If any sections are missing, add placeholders
    for header in section_headers:
        key = header.lower().replace(' ', '_')
        if key not in report_sections:
            report_sections[key] = f"No {header} content provided."
    
    # Debug: Print the extracted sections
    print("\nExtracted report sections:")
    for section, content in report_sections.items():
        print(f"Section: {section}")
        print(f"Content length: {len(content)} characters")
        if len(content) < 10:
            print(f"Warning: Section '{section}' has very little content!")
    
    charts = generate_charts(get_stock_data(stocks), pm_metrics)
    pdf_filename = f"Portfolio_Investment_Report_{current_date}.pdf"
    create_pdf_report(report_sections, charts, pm_metrics, pdf_filename)
    
except Exception as e:
    print(f"Error executing the crew: {e}")
    import traceback
    traceback.print_exc()

generate_workflow_diagram()

try:
    with open(f"Portfolio_Investment_Report_{current_date}.txt", "w") as f:
        f.write(str(result) if result else "Report generation failed")
    print(f"Text report saved to Portfolio_Investment_Report_{current_date}.txt")
except Exception as e:
    print(f"Error saving text report: {e}")
