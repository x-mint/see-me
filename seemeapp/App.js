import React, { Component } from 'react';
import {
  Image,
  Button,
  StyleSheet,
  View,
  Text
} from 'react-native';
import ImagePicker from 'react-native-image-picker';

export default class App extends Component {

  constructor() {
    super();

    this.state = {
      imageSource: null,
      loadingResponse: false,
      displayingResult: false
    };
  }

  handleChoosePhoto = () => {
    this.setState({ displayingResult: false });
    this.setState({ imageSource: null });
    const options = {
      maxHeight: 256,
      maxWidth: 1600
    }
    ImagePicker.showImagePicker(options, response => {
      if (response.uri) {
        this.setState({ imageSource: response });
      }
    })
  };

  createFormData = (photo, body) => {

    const data = new FormData();

    data.append("photo", {
      name: photo.fileName,
      type: photo.type,
      uri:
        Platform.OS === "android" ? photo.uri : photo.uri.replace("file://", "")
    });

    Object.keys(body).forEach(key => {
      data.append(key, body[key]);
    });

    return data;
  };

  handleUpload = async () => {
    this.setState({ loadingResponse: true });
    const response = await fetch('http://34.82.15.64:3000/api/upload', {
      method: 'POST',
      body: this.createFormData(this.state.imageSource, {})
    });
    if (response.ok) {
      let responseJSON = await response.json();
      let image = responseJSON['image'];
      //image = image.replace('undefined', 'data:image/jpeg;base64,');
      image = 'data:image/jpeg;base64,' + image;
      this.setState({ imageSource: image });
      this.setState({ displayingResult: true });
    } else {
      alert("Error occured");
    }
    this.setState({ loadingResponse: false });
  };

  render() {

    if ( this.state.loadingResponse ) {
      return (
        <View>
          <Text>Loading...</Text>
        </View>
      );
    }

    const { imageSource, displayingResult } = this.state;
    return (
      <View style={ styles.container }>
        { !imageSource && (
          <Image
            source={ require('./logo.jpg') }
            style={ styles.logo }
          />
          )
        }
        { imageSource && (
            <React.Fragment>
              <Image
                source={ { uri: (typeof imageSource === 'string' ? imageSource : imageSource.uri) } }
                style={ styles.avatar }
              />
              {
                !displayingResult && (
                  <Button title="Upload" onPress={ this.handleUpload } />
                )
              }
            </React.Fragment>
          )
        }
        <Button title="Take/Choose Photo" onPress={ this.handleChoosePhoto } />
      </View>
    );
  }
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center'
  },
  logo: {
    alignSelf: 'center',
    height: 150,
    width: 250,
    borderWidth: 1,
    borderRadius: 50
  },
  avatar: {
     width: 300,
     height: 300
  }
});
