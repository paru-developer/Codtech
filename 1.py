"""
Weather API Integration and Visualization Dashboard
Author: [Your Name]
Date: February 2026
Project: Internship Assignment - API Integration & Data Visualization

Description:
This program fetches weather data from OpenWeatherMap API and creates
visualizations using matplotlib and seaborn.
"""

import requests
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
import json

# Set up nice looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


def fetch_weather_data(city, api_key="demo"):
    """
    Fetch current weather data for a city
    
    Parameters:
        city (str): Name of the city
        api_key (str): API key for OpenWeatherMap
    
    Returns:
        dict: Weather data or None if failed
    """
    # Using demo mode for testing (you can add real API key later)
    if api_key == "demo":
        # Generate sample data for demonstration
        import random
        temp = random.uniform(15, 30)
        return {
            'name': city,
            'temp': round(temp, 1),
            'humidity': random.randint(40, 80),
            'pressure': random.randint(1000, 1020),
            'wind_speed': round(random.uniform(2, 10), 1),
            'description': random.choice(['Clear', 'Cloudy', 'Rainy'])
        }
    
    # Real API call (uncomment when you have API key)
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather"
        params = {
            'q': city,
            'appid': api_key,
            'units': 'metric'
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        return {
            'name': data['name'],
            'temp': data['main']['temp'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'wind_speed': data['wind']['speed'],
            'description': data['weather'][0]['description']
        }
    except Exception as e:
        print(f"Error fetching data for {city}: {e}")
        return None


def create_temperature_chart(weather_df):
    """Create a bar chart for temperature comparison"""
    plt.figure(figsize=(12, 6))
    
    # Create bar chart
    bars = plt.bar(weather_df['City'], weather_df['Temperature'], 
                   color='coral', edgecolor='black', alpha=0.7)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}°C',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.title('Temperature Comparison Across Cities', fontsize=16, fontweight='bold')
    plt.xlabel('City', fontsize=12)
    plt.ylabel('Temperature (°C)', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('temperature_chart.png', dpi=300, bbox_inches='tight')
    print("✓ Temperature chart saved!")
    plt.close()


def create_humidity_chart(weather_df):
    """Create a bar chart for humidity comparison"""
    plt.figure(figsize=(12, 6))
    
    bars = plt.bar(weather_df['City'], weather_df['Humidity'], 
                   color='skyblue', edgecolor='black', alpha=0.7)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.title('Humidity Comparison Across Cities', fontsize=16, fontweight='bold')
    plt.xlabel('City', fontsize=12)
    plt.ylabel('Humidity (%)', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('humidity_chart.png', dpi=300, bbox_inches='tight')
    print("✓ Humidity chart saved!")
    plt.close()


def create_combined_dashboard(weather_df):
    """Create a comprehensive dashboard with multiple plots"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Weather Dashboard - Multi-City Analysis', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    # Temperature plot
    axes[0, 0].bar(weather_df['City'], weather_df['Temperature'], 
                   color='orangered', alpha=0.7)
    axes[0, 0].set_title('Temperature (°C)', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Temperature (°C)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Humidity plot
    axes[0, 1].bar(weather_df['City'], weather_df['Humidity'], 
                   color='steelblue', alpha=0.7)
    axes[0, 1].set_title('Humidity (%)', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Humidity (%)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Pressure plot
    axes[1, 0].bar(weather_df['City'], weather_df['Pressure'], 
                   color='green', alpha=0.7)
    axes[1, 0].set_title('Pressure (hPa)', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Pressure (hPa)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Wind Speed plot
    axes[1, 1].bar(weather_df['City'], weather_df['Wind Speed'], 
                   color='purple', alpha=0.7)
    axes[1, 1].set_title('Wind Speed (m/s)', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Wind Speed (m/s)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('weather_dashboard.png', dpi=300, bbox_inches='tight')
    print("✓ Complete dashboard saved!")
    plt.close()


def create_correlation_heatmap(weather_df):
    """Create a heatmap showing correlations between weather parameters"""
    plt.figure(figsize=(10, 8))
    
    # Select numeric columns for correlation
    numeric_df = weather_df[['Temperature', 'Humidity', 'Pressure', 'Wind Speed']]
    
    # Calculate correlation
    correlation = numeric_df.corr()
    
    # Create heatmap
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=2, cbar_kws={"shrink": 0.8},
                fmt='.2f', vmin=-1, vmax=1)
    
    plt.title('Weather Parameters Correlation Analysis', 
              fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ Correlation heatmap saved!")
    plt.close()


def save_data_to_json(weather_df):
    """Save the weather data to a JSON file"""
    data_dict = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'cities_analyzed': len(weather_df),
        'data': weather_df.to_dict(orient='records')
    }
    
    with open('weather_data.json', 'w') as f:
        json.dump(data_dict, f, indent=4)
    
    print("✓ Data saved to JSON file!")


def main():
    """Main function to run the weather dashboard"""
    
    print("=" * 60)
    print("   WEATHER DATA VISUALIZATION DASHBOARD")
    print("=" * 60)
    print()
    
    # List of cities to analyze
    cities = ['London', 'New York', 'Tokyo', 'Paris', 'Sydney', 'Mumbai']
    
    print(f"Fetching weather data for {len(cities)} cities...")
    print(f"Cities: {', '.join(cities)}")
    print()
    
    # Fetch data for all cities
    weather_data = []
    for city in cities:
        print(f"  → Fetching data for {city}...")
        data = fetch_weather_data(city)
        if data:
            weather_data.append(data)
    
    # Create DataFrame
    weather_df = pd.DataFrame(weather_data)
    weather_df.columns = ['City', 'Temperature', 'Humidity', 'Pressure', 
                          'Wind Speed', 'Description']
    
    print()
    print("=" * 60)
    print("CURRENT WEATHER DATA")
    print("=" * 60)
    print(weather_df.to_string(index=False))
    print()
    
    # Create visualizations
    print("=" * 60)
    print("Creating visualizations...")
    print("=" * 60)
    
    create_temperature_chart(weather_df)
    create_humidity_chart(weather_df)
    create_combined_dashboard(weather_df)
    create_correlation_heatmap(weather_df)
    save_data_to_json(weather_df)
    
    print()
    print("=" * 60)
    print("✅ ALL VISUALIZATIONS COMPLETED!")
    print("=" * 60)
    print()
    print("Generated files:")
    print("  📊 temperature_chart.png")
    print("  📊 humidity_chart.png")
    print("  📊 weather_dashboard.png")
    print("  📊 correlation_heatmap.png")
    print("  💾 weather_data.json")
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
