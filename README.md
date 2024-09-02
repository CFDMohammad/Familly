    
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>My Simple Website</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f4f4f4;
            color: #333;
        }
        header {
            background-color: #4CAF50;
            color: white;
            padding: 10px 0;
            text-align: center;
        }
        nav {
            display: flex;
            justify-content: center;
            background-color: #333;
        }
        nav a {
            color: white;
            padding: 14px 20px;
            text-decoration: none;
            text-align: center;
        }
        nav a:hover {
            background-color: #575757;
        }
        .container {
            max-width: 1200px;
            margin: 20px auto;
            padding: 20px;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        footer {
            background-color: #333;
            color: white;
            text-align: center;
            padding: 10px 0;
            position: fixed;
            width: 100%;
            bottom: 0;
        }
    </style>
</head>
<body>
    <header>
        <h1>Welcome to My Website</h1>
    </header>
    <nav>
        <a href="#home">Home</a>
        <a href="#about">About</a>
        <a href="#services">Services</a>
        <a href="#contact">Contact</a>
    </nav>
    <div class="container">
        <section id="home">
            <h2>Home</h2>
            <p>This is the home section of the website. You can put some introductory content here.</p>
        </section>
        <section id="about">
            <h2>About</h2>
            <p>This section is about the website or the author. You can include details about yourself or your business here.</p>
        </section>
        <section id="services">
            <h2>Services</h2>
            <p>Describe the services you offer or the products you sell in this section.</p>
        </section>
        <section id="contact">
            <h2>Contact</h2>
            <p>Provide contact details or a contact form here.</p>
        </section>
    </div>
    <footer>
        <p>&copy; 2024 My Simple Website. All rights reserved.</p>
    </footer>
</body>
</html>
