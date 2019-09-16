# SeeMeApp

Connects to the server with port 8081.

### Usage

Go to instance and open ssh in browser
```
cd SeeMe/seeMeEnv
source bin/activate
```

Run the server
```
cd SeeMe/server
node index
```

After runnig the server open Android Studio Virtual Device. Then using command line:
```
cd SeeMeApp
npm start
```

Then close the command line and re-open again and go to the app directory.
Run:
```
react-native run-android
```

If necessary you can print the logs by the following command:
```
react-native log-android
```
