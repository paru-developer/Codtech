"""
Automated Report Generation Script
Author: [PARVATHI ARUN]
Date: February 2026
Project: Internship Assignment - Automated PDF Report Generation

Description:
This script reads data from a CSV file, performs analysis, and generates
a professional PDF report with charts and statistics.
"""

import pandas as pd
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from datetime import datetime
import os


def read_data(file_path):
    """
    Read data from CSV file
    
    Parameters:
        file_path (str): Path to the CSV file
    
    Returns:
        DataFrame: Pandas DataFrame with the data
    """
    try:
        df = pd.read_csv(file_path)
        print(f"✓ Data loaded successfully: {len(df)} rows")
        return df
    except Exception as e:
        print(f"Error reading file: {e}")
        return None


def analyze_data(df):
    """
    Perform basic analysis on the data
    
    Parameters:
        df (DataFrame): Input data
    
    Returns:
        dict: Dictionary containing analysis results
    """
    analysis = {}
    
    # Basic statistics
    analysis['total_sales'] = df['Sales'].sum()
    analysis['total_quantity'] = df['Quantity'].sum()
    analysis['average_sale'] = df['Sales'].mean()
    analysis['total_records'] = len(df)
    
    # Sales by category
    analysis['sales_by_category'] = df.groupby('Category')['Sales'].sum().to_dict()
    
    # Sales by region
    analysis['sales_by_region'] = df.groupby('Region')['Sales'].sum().to_dict()
    
    # Top products
    analysis['top_products'] = df.groupby('Product')['Sales'].sum().sort_values(ascending=False).head(5).to_dict()
    
    print("✓ Data analysis completed")
    return analysis


def create_charts(df):
    """
    Create visualization charts for the report
    
    Parameters:
        df (DataFrame): Input data
    
    Returns:
        list: List of chart file paths
    """
    chart_files = []
    
    # Chart 1: Sales by Category
    plt.figure(figsize=(8, 5))
    category_sales = df.groupby('Category')['Sales'].sum()
    plt.bar(category_sales.index, category_sales.values, color='skyblue', edgecolor='black')
    plt.title('Total Sales by Category', fontsize=14, fontweight='bold')
    plt.xlabel('Category')
    plt.ylabel('Sales ($)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('chart_category.png', dpi=150, bbox_inches='tight')
    chart_files.append('chart_category.png')
    plt.close()
    print("✓ Category chart created")
    
    # Chart 2: Sales by Region
    plt.figure(figsize=(8, 5))
    region_sales = df.groupby('Region')['Sales'].sum()
    colors_list = ['coral', 'lightgreen', 'gold', 'lightblue']
    plt.pie(region_sales.values, labels=region_sales.index, autopct='%1.1f%%', 
            colors=colors_list, startangle=90)
    plt.title('Sales Distribution by Region', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('chart_region.png', dpi=150, bbox_inches='tight')
    chart_files.append('chart_region.png')
    plt.close()
    print("✓ Region chart created")
    
    # Chart 3: Top 5 Products
    plt.figure(figsize=(8, 5))
    top_products = df.groupby('Product')['Sales'].sum().sort_values(ascending=False).head(5)
    plt.barh(top_products.index, top_products.values, color='lightcoral', edgecolor='black')
    plt.title('Top 5 Products by Sales', fontsize=14, fontweight='bold')
    plt.xlabel('Sales ($)')
    plt.ylabel('Product')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('chart_products.png', dpi=150, bbox_inches='tight')
    chart_files.append('chart_products.png')
    plt.close()
    print("✓ Products chart created")
    
    return chart_files


def generate_pdf_report(df, analysis, chart_files, output_file='Sales_Report.pdf'):
    """
    Generate a professional PDF report
    
    Parameters:
        df (DataFrame): Input data
        analysis (dict): Analysis results
        chart_files (list): List of chart file paths
        output_file (str): Output PDF file name
    """
    # Create PDF document
    doc = SimpleDocTemplate(output_file, pagesize=letter)
    elements = []
    
    # Get styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f4788'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#2e5090'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    # Title
    title = Paragraph("SALES ANALYSIS REPORT", title_style)
    elements.append(title)
    elements.append(Spacer(1, 0.2*inch))
    
    # Report metadata
    report_date = datetime.now().strftime("%B %d, %Y")
    meta_text = f"<b>Report Generated:</b> {report_date}<br/><b>Data Period:</b> January - February 2024"
    meta = Paragraph(meta_text, styles['Normal'])
    elements.append(meta)
    elements.append(Spacer(1, 0.3*inch))
    
    # Executive Summary
    elements.append(Paragraph("EXECUTIVE SUMMARY", heading_style))
    
    summary_text = f"""
    This report provides a comprehensive analysis of sales data 
    """
    elements.append(Paragraph(summary_text, styles['BodyText']))
    elements.append(Spacer(1, 0.3*inch))
    
    # Key Metrics Table
    elements.append(Paragraph("KEY PERFORMANCE METRICS", heading_style))
    
    metrics_data = [
        ['Metric', 'Value'],
        ['Total Revenue', f'${analysis["total_sales"]:,.2f}'],
        ['Total Units Sold', f'{analysis["total_quantity"]:,}'],
        ['Average Sale Value', f'${analysis["average_sale"]:.2f}'],
        ['Total Transactions', f'{analysis["total_records"]}'],
    ]
    
    metrics_table = Table(metrics_data, colWidths=[3*inch, 2*inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4788')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
    ]))
    
    elements.append(metrics_table)
    elements.append(Spacer(1, 0.4*inch))
    
    # Sales by Category
    elements.append(Paragraph("SALES BY CATEGORY", heading_style))
    
    category_data = [['Category', 'Total Sales']]
    for category, sales in analysis['sales_by_category'].items():
        category_data.append([category, f'${sales:,.2f}'])
    
    category_table = Table(category_data, colWidths=[3*inch, 2*inch])
    category_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2e5090')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
    ]))
    
    elements.append(category_table)
    elements.append(Spacer(1, 0.3*inch))
    
    # Add chart
    if len(chart_files) > 0 and os.path.exists(chart_files[0]):
        img = Image(chart_files[0], width=5*inch, height=3*inch)
        elements.append(img)
    
    elements.append(PageBreak())
    
    # Sales by Region
    elements.append(Paragraph("SALES BY REGION", heading_style))
    
    region_data = [['Region', 'Total Sales']]
    for region, sales in analysis['sales_by_region'].items():
        region_data.append([region, f'${sales:,.2f}'])
    
    region_table = Table(region_data, colWidths=[3*inch, 2*inch])
    region_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2e5090')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
    ]))
    
    elements.append(region_table)
    elements.append(Spacer(1, 0.3*inch))
    
    # Add region chart
    if len(chart_files) > 1 and os.path.exists(chart_files[1]):
        img = Image(chart_files[1], width=5*inch, height=3*inch)
        elements.append(img)
    
    elements.append(Spacer(1, 0.3*inch))
    
    # Top Products
    elements.append(Paragraph("TOP 5 PRODUCTS", heading_style))
    
    # Add products chart
    if len(chart_files) > 2 and os.path.exists(chart_files[2]):
        img = Image(chart_files[2], width=5*inch, height=3*inch)
        elements.append(img)
    
    elements.append(Spacer(1, 0.4*inch))
    
    # Conclusion
    elements.append(Paragraph("CONCLUSION", heading_style))
    conclusion_text = """
    The analysis reveals strong sales performance across all regions with Electronics being the 
    dominant category. Laptop sales show the highest revenue contribution among individual products. 
    The data suggests opportunities for growth in the Furniture category and consistent performance 
    across all geographic regions.
    """
    elements.append(Paragraph(conclusion_text, styles['BodyText']))
    
    # Footer
    elements.append(Spacer(1, 0.5*inch))
    footer_text = f"<i>Report generated automatically on {report_date}</i>"
    elements.append(Paragraph(footer_text, styles['Italic']))
    
    # Build PDF
    doc.build(elements)
    print(f"✓ PDF report generated: {output_file}")


def main():
    """Main function to run the automated report generation"""
    
    print("=" * 60)
    print("   AUTOMATED REPORT GENERATION SYSTEM")
    print("=" * 60)
    print()
    
    # Step 1: Read data
    print("Step 1: Reading data from file...")
    df = read_data('sales_data.csv')
    
    if df is None:
        print("Failed to read data. Exiting.")
        return
    
    print()
    print("Data Preview:")
    print(df.head())
    print()
    
    # Step 2: Analyze data
    print("Step 2: Analyzing data...")
    analysis = analyze_data(df)
    print()
    
    # Display analysis summary
    print("Analysis Summary:")
    print(f"  Total Sales: ${analysis['total_sales']:,.2f}")
    print(f"  Total Quantity: {analysis['total_quantity']}")
    print(f"  Average Sale: ${analysis['average_sale']:.2f}")
    print()
    
    # Step 3: Create charts
    print("Step 3: Creating visualization charts...")
    chart_files = create_charts(df)
    print()
    
    # Step 4: Generate PDF report
    print("Step 4: Generating PDF report...")
    generate_pdf_report(df, analysis, chart_files)
    print()
    
    print("=" * 60)
    print("✅ REPORT GENERATION COMPLETED!")
    print("=" * 60)
    print()
    print("Generated files:")
    print("  📄 Sales_Report.pdf - Complete analytical report")
    print("  📊 chart_category.png - Category sales chart")
    print("  📊 chart_region.png - Regional distribution chart")
    print("  📊 chart_products.png - Top products chart")
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
