/**
 * Sample React Native App
 * https://github.com/facebook/react-native
 * @flow
 */

import React, { Component } from 'react';
import {
  View,
  StyleSheet,
  Dimensions,
  TouchableHighlight,
  TouchableOpacity,
  ImageBackground,
  Image,
  Text,
  ActivityIndicator,
  Linking,
  Alert,
} from 'react-native';

import { RNCamera } from 'react-native-camera'
import RNFetchBlob from 'react-native-fetch-blob'
// import axios from 'axios'
import Amplify, { Storage } from 'aws-amplify';
import aws_exports from './aws-exports';
import { Buffer } from 'buffer'
import axios from 'axios';


// var paperspace_node = require('paperspace-node');
// var paperspace = paperspace_node({
//   apiKey: 'b111c6bbfde0bc5a8357eada913762' // <- paste your api key here
// });

console.disableYellowBox = true;
Amplify.configure(aws_exports);
console.log("Config success")

const styles = StyleSheet.create({
  loading: {
    flex: 1,
    justifyContent: 'center',
    position: 'absolute',
    left: 0,
    right: 0,
    top: 0,
    bottom: 0,
    alignItems: 'center',
    justifyContent: 'center'
  },
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#000',
  },
  horizontal: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    padding: 10
  },
  preview: {
    flex: 1,

    justifyContent: 'flex-end',
    alignItems: 'center',
    height: Dimensions.get('window').height,
    width: Dimensions.get('window').width
  },
  disabledButton: {
    backgroundColor: 'red',
    borderColor: 'white',
    borderWidth: 1,
    borderRadius: 12,
    color: 'white',
    fontSize: 24,
    fontWeight: 'bold',
    overflow: 'hidden',
    padding: 12,
    textAlign:'center',
  },
  recordIndicatorShown: {
    color: 'red',
    fontSize: 18,
    position: 'absolute',
    right:     40,
    top:      40,
  },
  recordIndicatorHidden: {
    color: 'red',
    fontSize: 18,
    position: 'absolute',
    right:     40,
    top:      40,
    display: 'none',
  },
  watchButton: {
    backgroundColor: '#16afe2',
    borderColor: 'white',
    borderWidth: 1,
    borderRadius: 12,
    color: 'white',
    fontSize: 24,
    fontWeight: 'bold',
    overflow: 'hidden',
    padding: 12,
    textAlign:'center',
  },
  capture: {
   backgroundColor: '#16afe2',
    borderColor: 'white',
    borderWidth: 1,
    borderRadius: 12,
    color: 'white',
    fontSize: 24,
    fontWeight: 'bold',
    overflow: 'hidden',
    padding: 12,
    textAlign:'center',
  }, switch: {
    position: 'absolute',
    width: 30,
    height: 30,
    borderRadius: 35,
    borderWidth: 3,
    borderColor: '#FFF',
    top: 30,
    right: 30,
  },
  cancel: {
    position: 'absolute',
    right: 20,
    top: 20,
    backgroundColor: 'transparent',
    color: '#FFF',
    fontWeight: 'bold',
    fontSize: 17,
  }, label: {
    width: 500,
    height: 500,
    color: 'red',

    textAlign: 'center',
    fontSize: 80,
    justifyContent: "center",
    alignItems: "center"
  }
});

export default class App extends Component {

  constructor() {
    super();
    this.state = {
      videoReady: false
    }
    
  }



  async startRecording() {
    this.setState({ recording: true });
    this.setState({ videoReady: false });

    // delete previous video file 
   
    await Storage.remove("stylized.mp4")
           
          
    // default to mp4 for android as codec is not set
    const { uri, codec = "mp4" } = await this.camera.recordAsync();
    this.setState({ recording: false, processing: true });



    const dirs = RNFetchBlob.fs.dirs
    let arr = dirs.CacheDir.split('/')
    let parentDir = arr.slice(0, -1).join("/")
    arr = uri.split('/')
    filePath = `${parentDir}/CachesCamera/${arr[arr.length - 1]}`


    const data = await RNFetchBlob.fs.readFile(filePath, 'base64');
    const fileObject = new Buffer(data, 'base64');
    const access = {
      level: "public",
      ACL: 'public-read'
    }
    let key = "testvideo.mp4"
    await Storage.put(key, fileObject, access)
    // apiKey: 'b111c6bbfde0bc5a8357eada913762' 
    let headers = {
      "x-api-key": '***********',
      "Content-Type": "application/json"
    }
    let body = {
      "container": "garfieldchh/snap_prod:latest",
      "machineType": "GPU+",
      "command": "cd /storage/snap_prod  && python style.py",
      "project": "snapmobile"
    }

    let paperspace_endpoint = "https://api.paperspace.io/jobs/createJob"

    await axios.post(paperspace_endpoint, body, { headers: headers })
      .then(res => {
        console.log("start job runner")
        this.setState({ processing: false })
      })
      .catch(err => console.error(e))


    // Let the user know that the file is being procceseed 
    Alert.alert('video is being proccessed. The watch button will turn blue when ready')
    
    setInterval(async () => {
      
      if (!this.state.videoReady){
        await Storage.list("")
          .then(res => {
           
            let ready = false
  
            res.forEach(function(item) {
             
              if(item["key"] == "stylized.mp4"){
                ready = true 
              }
            });
            if (ready){
              this.setState({videoReady: true})
              Alert.alert("video is ready")
            }

          })
          .catch(err => console.error(e))

      }
    }, 60000);
    

  }



  // file:///var/mobile/Containers/Data/Application/CAF25BE3-5EEA-4BA9-8015-7C231FEE85D2/Library/CachesCamera/

  openVideo() {
    // let app = this
    // console.log(this.state.videoReady)
    // console.log(app.videoReady)
    // console.log(videoReady)
    // console.log("open video called")
    if(!this.state.videoReady){
      Alert.alert("video is not available")
      return 
    }
    

    const url = `https://s3.amazonaws.com/artsnap-userfiles-mobilehub-1207532684/public/stylized.mp4`;

    Linking.openURL(url).catch(err =>
      console.error("An error occurred opening the link", err)
    );
  }
  stopRecording() {
    this.camera.stopRecording();
    console.log("stop recording")
  }

  render() {
    const { recording, processing } = this.state;

    let button = (
      <TouchableOpacity
        onPress={this.startRecording.bind(this)}
        style={styles.capture}
      >
        <Text style={{ color: 'white', fontSize: 14 }}>Record </Text>
      </TouchableOpacity>
    );

    let watch_button = (
      <TouchableHighlight
        underlayColor="rgba(200,200,200,0.6)"
        // disabled={!this.state.videoReady}
        // disabled={false}
        onPress={this.openVideo.bind(this)}
        // style={styles.watchButton}
        style={!this.state.videoReady ? styles.disabledButton: styles.watchButton}
       
      >
        <Text style={{ color: 'white', fontSize: 14 }}>Watch </Text>
      </TouchableHighlight>
    );

    if (recording) {
      button = (
        <TouchableOpacity
          onPress={this.stopRecording.bind(this)}
          style={styles.capture}
        >
          <Text style={{ fontSize: 14 }}> stop </Text>
        </TouchableOpacity>
      );
    }

    if (processing) {
      button = (
        <View style={styles.capture}>
          <ActivityIndicator animating />
        </View>
      );
    }

    return (
      <View style={styles.container}>
        <RNCamera
          ref={ref => {
            this.camera = ref;
          }}
          style={styles.preview}
          type={RNCamera.Constants.Type.back}
          flashMode={RNCamera.Constants.FlashMode.on}
          permissionDialogTitle={"Permission to use camera"}
          permissionDialogMessage={
            "We need your permission to use your camera phone"
          }
        />
         {/* style={!this.state.videoReady ? styles.disabledButton: styles.watchButton} */}
        <Text style={this.state.recording ? styles.recordIndicatorShown: styles.recordIndicatorHidden}>
          {"Recording"}
        </Text>
        <View
          style={{ flex: 0, flexDirection: "row", justifyContent: "center" }}
        >
          {button}
          {watch_button}
        </View>
      </View>
    );
  }


}

