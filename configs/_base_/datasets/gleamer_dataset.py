dataset_info = dict(
    dataset_name='gleamer',
    paper_info=dict(
        author='ZHANG Zekun',
        title='Keypoint dataset for Gleamer',
        year="2022",
        homepage="https://www.gleamer.ai/"
    ),
    keypoint_info={
        0: dict(name='CALCANEUS_POST', id=0, color=[51, 153, 255], swap=''),
        1: dict(name='SOMMET', id=0, color=[51, 153, 255], swap=''),
        2: dict(name='GRANDE_TUBEROSITE', id=0, color=[51, 153, 255], swap=''),
        3: dict(name='TALUS_COL', id=0, color=[0, 255, 0], swap=''),
        4: dict(name='TALUS_DISTAL', id=0, color=[0, 255, 0], swap=''),
        5: dict(name='M_1_PROXIMAL', id=0, color=[0, 255, 0], swap=''),
        6: dict(name='M_1_DISTAL', id=0, color=[0, 255, 0], swap=''),
        7: dict(name='CALCANEUS_ANT', id=0, color=[255, 128, 0], swap=''),
    },
    skeleton_info={
        0:
            dict(link=('CALCANEUS_POST', 'SOMMET'), id=0, color=[51, 153, 255]),
        1:
            dict(link=('SOMMET', 'GRANDE_TUBEROSITE'), id=1, color=[51, 153, 255]),
        2:
            dict(link=('TALUS_COL', 'TALUS_DISTAL'), id=2, color=[0, 255, 0]),
        3:
            dict(link=('TALUS_DISTAL', 'M_1_PROXIMAL'), id=3, color=[0, 255, 0]),
        4:
            dict(link=('M_1_PROXIMAL', 'M_1_DISTAL'), id=4, color=[0, 255, 0]),
        5:
            dict(link=('CALCANEUS_ANT', 'CALCANEUS_POST'), id=5, color=[255, 128, 0]),
    },
    joint_weights=[
        1., 1., 1., 1., 1., 1., 1., 1.
    ],
    sigmas=[
        0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025,
    ])
